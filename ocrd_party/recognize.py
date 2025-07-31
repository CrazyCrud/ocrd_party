import os
import uuid
from functools import cached_property
from typing import Optional
import logging

import torch
import numpy as np
from PIL import Image
from lxml import etree

from ocrd import Processor, OcrdPage, OcrdPageResult
from ocrd_models.ocrd_page import (
    TextEquivType,
    CoordsType,
    WordType,
    GlyphType
)
from ocrd_utils import (
    getLogger,
    points_from_polygon,
    coordinates_for_segment,
    polygon_from_x0y0x1y1,
    coordinates_of_segment,
    VERSION as OCRD_VERSION
)

from party.fusion import PartyModel
from party.pred import batched_pred
from lightning.fabric import Fabric


class PartyRecognize(Processor):
    """
    An OCR-D Processor that uses Party to recognize text on each TextLine.
    """

    @property
    def executable(self):
        return 'ocrd-party-recognize'

    def show_version(self):
        import party
        print(f"ocrd-party {self.version}, party {party.__version__}, ocrd/core {OCRD_VERSION}")

    def setup(self):
        """
        Load the Party model once and prepare Fabric.
        """
        # Model loading
        model_path = self.parameter['model_path']
        if not model_path:
            # Could use default HTRMoPo model
            model_path = self.parameter.get('model_htr_mopo', '10.5281/zenodo.14616981')
            if model_path:
                from htrmopo import get_model
                model_path = get_model(model_path) / 'model.safetensors'

        self.logger.info(f"Loading Party model from {model_path}")
        self.model = PartyModel.from_safetensors(model_path)

        # Device setup
        device = self.parameter.get('device', 'cpu')
        if device.startswith('cuda') and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available, falling back to CPU")
            device = 'cpu'

        # Precision setup
        precision = self.parameter.get('precision', '32-true')

        # Initialize Fabric
        self.fabric = Fabric(
            accelerator='gpu' if device.startswith('cuda') else 'cpu',
            devices=1,
            precision=precision
        )

        # Setup model with Fabric
        self.model = self.fabric.setup(self.model)

        # Compilation if requested
        if self.parameter.get('compile', False):
            try:
                self.logger.info("Compiling model with torch.compile")
                self.model = torch.compile(self.model, mode='max-autotune')
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")

        # Quantization if requested
        if self.parameter.get('quantize', False):
            try:
                import torchao
                self.logger.info("Quantizing model")
                torchao.quantization.utils.recommended_inductor_config_setter()
                self.model = torchao.autoquant(self.model)
            except Exception as e:
                self.logger.warning(f"Model quantization failed: {e}")

        self.batch_size = self.parameter.get('batch_size', 4)
        self.add_lang_token = self.parameter.get('add_lang_token', False)
        self.textequiv_level = self.parameter.get('textequiv_level', 'line')
        self.glyph_conf_cutoff = self.parameter.get('glyph_conf_cutoff', 0.1)

        # Features for image extraction
        self.features = self.parameter.get('features', '')

    @cached_property
    def _prompt_mode(self):
        """Get the prompt mode from the model."""
        return getattr(self.model, 'line_prompt_mode', 'curves')

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """
        Perform text recognition with Party.
        """
        pcgts = input_pcgts[0]
        page = pcgts.get_Page()

        # Get page image
        page_image, page_coords, page_image_info = self.workspace.image_from_page(
            page, page_id, feature_selector=self.features
        )

        # Collect all text lines from all text regions
        all_lines = []
        for region in page.get_AllRegions(classes=["Text"]):
            region_image, region_coords = self.workspace.image_from_segment(
                region, page_image, page_coords, feature_selector=self.features
            )

            textlines = region.get_TextLine()
            self.logger.info(f"About to recognize {len(textlines)} lines of region '{region.id}'")

            for line in textlines:
                # Get line image and coordinates
                line_image, line_coords = self.workspace.image_from_segment(
                    line, region_image, region_coords, feature_selector=self.features
                )

                # Skip empty or too small lines
                if (not all(line_image.size) or
                        line_image.height <= 8 or
                        line_image.width <= 8):
                    self.logger.warning(f"Skipping empty/tiny line '{line.id}' in region '{region.id}'")
                    continue

                all_lines.append({
                    'line': line,
                    'image': line_image,
                    'coords': line_coords,
                    'region_id': region.id
                })

        if not all_lines:
            self.logger.warning(f"No valid text lines on page '{page_id}'")
            return OcrdPageResult(pcgts)

        # Build Kraken container for Party
        from kraken.containers import Segmentation, BaselineLine

        baselines = []
        line_mapping = {}  # Map BaselineLine ID to original line object

        for idx, line_data in enumerate(all_lines):
            line = line_data['line']
            line_coords = line_data['coords']

            # Get baseline or fall back to polygon
            if line.get_Baseline():
                baseline_points = [(pt.x, pt.y) for pt in line.get_Baseline().points]
            else:
                # Create pseudo-baseline from polygon center line
                coords = line.get_Coords()
                if coords:
                    polygon = [(pt.x, pt.y) for pt in coords.points]
                    # Simple center line: top-center to bottom-center
                    xs = [p[0] for p in polygon]
                    ys = [p[1] for p in polygon]
                    x_center = (min(xs) + max(xs)) / 2
                    baseline_points = [(x_center, min(ys)), (x_center, max(ys))]
                else:
                    self.logger.warning(f"Line '{line.id}' has no baseline or coords, skipping")
                    continue

            # Convert to page coordinates
            baseline_page = coordinates_for_segment(
                np.array(baseline_points), None, line_coords
            )

            # Get boundary polygon
            if line.get_Coords():
                boundary = [(pt.x, pt.y) for pt in line.get_Coords().points]
                boundary_page = coordinates_for_segment(
                    np.array(boundary), None, line_coords
                )
            else:
                boundary_page = baseline_page

            # Create BaselineLine for Party
            bl_id = f"line_{idx}"
            bl = BaselineLine(
                id=bl_id,
                baseline=baseline_page.tolist(),
                boundary=boundary_page.tolist(),
                tags=[]
            )
            baselines.append(bl)
            line_mapping[bl_id] = line_data

        # Create segmentation container
        segmentation = Segmentation(
            text_direction=self.parameter.get('text_direction', 'horizontal-lr'),
            imagename=page_id or 'page',
            type='baselines' if self._prompt_mode == 'curves' else 'bbox',
            lines=baselines,
            regions={},
            script_detection=False,
            line_orders=[]
        )

        # Set languages if specified
        languages = self.parameter.get('languages', [])
        if languages:
            segmentation.language = languages

        # Run prediction
        self.logger.info(f"Running Party prediction on {len(baselines)} lines")

        with torch.inference_mode():
            pred_iter = batched_pred(
                model=self.model,
                im=page_image,
                bounds=segmentation,
                fabric=self.fabric,
                prompt_mode=self._prompt_mode,
                batch_size=self.batch_size,
                add_lang_token=self.add_lang_token
            )

            # Process predictions
            for pred in pred_iter:
                if pred.line.id not in line_mapping:
                    self.logger.warning(f"Prediction for unknown line ID: {pred.line.id}")
                    continue

                line_data = line_mapping[pred.line.id]
                line = line_data['line']
                line_coords = line_data['coords']

                # Clear existing text
                if line.get_TextEquiv():
                    self.logger.warning(f"Line '{line.id}' already contained text results")
                line.set_TextEquiv([])

                # Calculate average confidence
                if pred.confidences:
                    avg_conf = float(sum(pred.confidences) / len(pred.confidences))
                else:
                    avg_conf = 0.0

                # Set line text
                line.set_TextEquiv([
                    TextEquivType(Unicode=pred.prediction, conf=avg_conf)
                ])

                # Add language info to custom field if detected
                if hasattr(pred, 'language') and pred.language:
                    custom = line.get_custom() or ''
                    if not custom:
                        custom = f"language: {','.join(pred.language)}"
                    else:
                        custom += f"; language: {','.join(pred.language)}"
                    line.set_custom(custom)

                # Handle word segmentation if requested
                if self.textequiv_level in ['word', 'glyph']:
                    self._segment_into_words(
                        line, pred, line_data['image'].height, line_coords
                    )

                self.logger.debug(f"Recognized line '{line.id}': {pred.prediction[:50]}...")

        # Update higher-level text equivalences
        _page_update_higher_textequiv_levels(self.textequiv_level, pcgts)

        return OcrdPageResult(pcgts)

    def _segment_into_words(self, line, pred, line_height, line_coords):
        """
        Segment line into words based on spaces in prediction.
        """
        # Clear existing words
        if line.get_Word():
            self.logger.warning(f"Line '{line.id}' already contained word segmentation")
        line.set_Word([])

        # Simple word segmentation based on spaces
        text = pred.prediction
        if not text.strip():
            return

        # Calculate approximate character width
        line_width = line_coords['transform'][0, 0]  # Approximate from transform
        if hasattr(pred, 'positions') and pred.positions:
            # If Party provides character positions, use them
            positions = pred.positions
        else:
            # Otherwise, distribute characters evenly
            char_width = line_width / max(len(text), 1)
            positions = [i * char_width for i in range(len(text) + 1)]

        # Split into words
        words = []
        current_word = ""
        word_start = 0

        for i, char in enumerate(text):
            if char == ' ':
                if current_word:
                    words.append({
                        'text': current_word,
                        'start': word_start,
                        'end': i
                    })
                    current_word = ""
                word_start = i + 1
            else:
                current_word += char

        # Add last word
        if current_word:
            words.append({
                'text': current_word,
                'start': word_start,
                'end': len(text)
            })

        # Create Word elements
        for word_no, word_info in enumerate(words):
            if word_info['end'] <= word_info['start']:
                continue

            # Estimate word polygon
            if len(positions) > word_info['end']:
                x_start = positions[word_info['start']]
                x_end = positions[word_info['end']]
            else:
                # Fallback to proportional positioning
                x_start = (word_info['start'] / len(text)) * line_width
                x_end = (word_info['end'] / len(text)) * line_width

            polygon = polygon_from_x0y0x1y1([x_start, 0, x_end, line_height])
            points = points_from_polygon(
                coordinates_for_segment(polygon, None, line_coords)
            )

            word = WordType(
                id=f"{line.id}_word{word_no:04d}",
                Coords=CoordsType(points)
            )
            word.add_TextEquiv(TextEquivType(Unicode=word_info['text']))

            line.add_Word(word)


def _page_update_higher_textequiv_levels(level, pcgts):
    """
    Update the TextEquivs of all higher PAGE-XML hierarchy levels for consistency.
    """
    regions = pcgts.get_Page().get_TextRegion()
    if level != "region":
        for region in regions:
            lines = region.get_TextLine()
            if level != "line":
                for line in lines:
                    words = line.get_Word()
                    if level != "word":
                        for word in words:
                            glyphs = word.get_Glyph()
                            word_unicode = "".join(
                                glyph.get_TextEquiv()[0].Unicode
                                if glyph.get_TextEquiv()
                                else ""
                                for glyph in glyphs
                            )
                            word.set_TextEquiv([TextEquivType(Unicode=word_unicode)])

                    line_unicode = " ".join(
                        word.get_TextEquiv()[0].Unicode if word.get_TextEquiv() else ""
                        for word in words
                    )
                    line.set_TextEquiv([TextEquivType(Unicode=line_unicode)])

            region_unicode = "\n".join(
                line.get_TextEquiv()[0].Unicode if line.get_TextEquiv() else ""
                for line in lines
            )
            region.set_TextEquiv([TextEquivType(Unicode=region_unicode)])
