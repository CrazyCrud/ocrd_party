import os
from functools import cached_property
from typing import Optional
import re

import torch
import numpy as np

from ocrd.processor.base import OcrdPageResult
from ocrd import Processor
from ocrd_models.ocrd_page import (
    BaselineType,
    OcrdPage,
    TextEquivType,
    CoordsType,
    WordType,
)
from ocrd_utils import (
    points_from_polygon,
    coordinates_for_segment,
    polygon_from_x0y0x1y1,
    polygon_from_points,
    coordinates_of_segment,  # already there
    bbox_from_polygon,  # NEW
    transform_coordinates,  # NEW
    VERSION as OCRD_VERSION,
)

from kraken.containers import Segmentation, BaselineLine
from party.fusion import PartyModel
from party.pred import batched_pred


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

        parameter = dict(self.parameter)

        # mirror kraken: required parameter "model"
        model_arg = parameter['model']  # required, so [] is OK here
        model_path = self.resolve_resource(model_arg)

        # accept either a directory (containing model.safetensors) or a direct file path
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, 'model.safetensors')

        model_path = str(model_path)

        self.logger.info(f"Loading Party model from {model_path}")
        self.model = PartyModel.from_safetensors(model_path)

        # Device setup
        device = parameter.get('device', 'cpu')
        if device.startswith('cuda') and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available, falling back to CPU")
            device = 'cpu'

        # map 'autocast' to Fabric precision, if you keep that name
        autocast = parameter.get('autocast', False)
        precision = '16-mixed' if autocast else '32-true'

        self.language = parameter.get('language', '').strip()

        from lightning.fabric import Fabric
        self.fabric = Fabric(
            accelerator='gpu' if device.startswith('cuda') else 'cpu',
            devices=1,
            precision=precision
        )
        self.model = self.fabric.setup(self.model)

        if parameter.get('compile', False):
            try:
                self.logger.info("Compiling model with torch.compile")
                self.model = torch.compile(self.model, mode='max-autotune')
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")

        if parameter.get('quantize', False):
            try:
                import torchao
                self.logger.info("Quantizing model")
                torchao.quantization.utils.recommended_inductor_config_setter()
                self.model = torchao.autoquant(self.model)
            except Exception as e:
                self.logger.warning(f"Model quantization failed: {e}")

        self.batch_size = parameter.get('batch_size', 4)
        self.add_lang_token = bool(parameter.get('add_lang_token', bool(self.language)))
        self.textequiv_level = self.parameter.get('textequiv_level', 'line')
        if self.textequiv_level not in ('line', 'word'):
            self.logger.warning(
                "Unsupported textequiv_level '%s' requested. Falling back to 'line'. "
                "Glyph-level output is not supported in ocrd-party-recognize.",
                self.textequiv_level
            )
            self.textequiv_level = 'line'
        self.features = parameter.get('features', '')

        self.debug_dir = parameter.get('debug_dir', '')
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            self.logger.info("Debug crops will be written to '%s'", self.debug_dir)

    @cached_property
    def _prompt_mode(self):
        """Get the prompt mode from the model."""
        mdl = getattr(self.model, 'module', self.model)
        return getattr(mdl, 'line_prompt_mode', 'curves')

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
                    "line": line,
                    "image": line_image,
                    "coords": line_coords,
                    "region_id": region.id,
                })

        if not all_lines:
            self.logger.warning(f"No valid text lines on page '{page_id}'")
            return OcrdPageResult(pcgts)

        baselines = []
        active_lines = []
        line_mapping = {}

        for line_data in all_lines:
            line = line_data["line"]

            # Get boundary polygon in IMAGE coords (same as page_image)
            poly_img = np.array(
                coordinates_of_segment(line, None, page_coords),
                dtype=float,
            )
            if poly_img.size == 0:
                self.logger.warning("Line '%s' has empty polygon, skipping", line.id)
                continue

            # Compute baseline in IMAGE coords
            if line.get_Baseline():
                # baseline defined in PAGE coords -> transform to image coords
                base = np.array(polygon_from_points(line.get_Baseline().points), dtype=float)
                base_img = transform_coordinates(base, page_coords["transform"])
            else:
                # fallback: pseudo-baseline near the bottom of the line bbox (in image coords)
                xmin, ymin, xmax, ymax = bbox_from_polygon(poly_img)
                ymid = ymin + 0.8 * (ymax - ymin)
                base_img = np.array([[xmin, ymid], [xmax, ymid]], dtype=float)

                # ALSO write this dummy baseline back into PAGE coords so
                # kraken/party CLI can see it later.
                base_page = coordinates_for_segment(base_img, None, page_coords)
                line.set_Baseline(
                    BaselineType(points=points_from_polygon(base_page))
                )

            baseline_page = [(float(x), float(y)) for x, y in base_img]
            boundary_page = [poly_img.tolist()]

            # store bbox for later cropping (debug)
            xmin, ymin, xmax, ymax = bbox_from_polygon(poly_img)
            line_data["bbox_img"] = (xmin, ymin, xmax, ymax)

            bl_id = line.id
            bl = BaselineLine(
                id=bl_id,
                baseline=baseline_page,
                boundary=boundary_page,
                tags=[],
            )
            baselines.append(bl)
            active_lines.append(line_data)
            line_mapping[bl_id] = line_data

        # Create segmentation container
        segmentation = Segmentation(
            text_direction=self.parameter.get('text_direction', 'horizontal-lr'),
            imagename=page_id or 'page',
            type='baseline',
            lines=baselines,
            regions={},
            script_detection=False,
            line_orders=[],
        )

        # Set language if specified
        if self.language:
            segmentation.language = self.language

        # Run prediction
        self.logger.info(f"Running Party prediction on {len(baselines)} lines")

        with torch.inference_mode():
            pred_iter = batched_pred(
                model=self.model,
                im=page_image,
                bounds=segmentation,
                fabric=self.fabric,
                batch_size=self.batch_size,
                add_lang_token=self.add_lang_token,
            )

            # Process predictions
            for idx, pred in enumerate(pred_iter):
                # 1st choice: use pred.line.id (Party keeps the segmentation line there)
                pred_line = getattr(pred, "line", None)
                pred_line_id = getattr(pred_line, "id", None) if pred_line is not None else None

                if pred_line_id in line_mapping:
                    line_data = line_mapping[pred_line_id]
                else:
                    # Fallback: rely on order if for some reason pred.line is missing
                    if idx >= len(active_lines):
                        self.logger.warning(
                            "Received more predictions (%d) than lines (%d)",
                            idx + 1, len(active_lines)
                        )
                        break
                    self.logger.warning(
                        "Prediction without usable line id at index %d; falling back to index mapping",
                        idx,
                    )
                    line_data = active_lines[idx]

                line = line_data["line"]
                line_coords = line_data["coords"]

                # Clear existing text
                if line.get_TextEquiv():
                    self.logger.warning(
                        "Line '%s' already contained text results", line.id
                    )
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
                if hasattr(pred, "language") and pred.language:
                    custom = line.get_custom() or ""
                    if not custom:
                        custom = f"language: {','.join(pred.language)}"
                    else:
                        custom += f"; language: {','.join(pred.language)}"
                    line.set_custom(custom)

                # Handle word segmentation if requested
                if self.textequiv_level == "word":
                    self._segment_into_words(
                        line=line,
                        pred=pred,
                        line_height=line_data["image"].height,
                        line_width=line_data["image"].width,
                        line_coords=line_coords,
                    )

                if self.debug_dir:
                    bbox = line_data.get("bbox_img")
                    if bbox is not None:
                        xmin, ymin, xmax, ymax = bbox
                        # pad a bit for nicer context
                        pad = 4
                        x0 = max(int(xmin) - pad, 0)
                        y0 = max(int(ymin) - pad, 0)
                        x1 = min(int(xmax) + pad, page_image.width)
                        y1 = min(int(ymax) + pad, page_image.height)

                        crop = page_image.crop((x0, y0, x1, y1))

                        safe_id = _sanitize_filename(line.id or f"line_{idx}")
                        # keep the text short in filename
                        short_text = _sanitize_filename(pred.prediction[:30])
                        filename = f"{page_id or 'page'}_{safe_id}_{idx:04d}_{short_text}.png"
                        out_path = os.path.join(self.debug_dir, filename)
                        try:
                            crop.save(out_path)
                        except Exception as e:
                            self.logger.warning(
                                "Failed to save debug crop for line '%s' to '%s': %s",
                                line.id, out_path, e
                            )
                    else:
                        self.logger.debug(
                            "No bbox_img stored for line '%s', cannot write debug crop",
                            line.id,
                        )

                self.logger.debug(
                    "Recognized line '%s': %s",
                    line.id,
                    pred.prediction[:50] + ("..." if len(pred.prediction) > 50 else ""),
                )

        # Update higher-level text equivalences
        _page_update_higher_textequiv_levels(self.textequiv_level, pcgts)

        return OcrdPageResult(pcgts)

    def _segment_into_words(self, line, pred, line_height, line_width, line_coords):
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
        line_width = line_coords.get('width', line_width)
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


def _points_to_list(points):
    """
    Normalize PAGE points (string/list) to a list of (x, y) tuples.
    Works whether polygon_from_points returns a list or a numpy array.
    """
    poly = polygon_from_points(points)
    # poly can be a numpy array or a list of [x, y]
    return [(float(p[0]), float(p[1])) for p in poly]


def _sanitize_filename(s: str) -> str:
    """
    Make a safe filename fragment from a line id or text.
    """
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^A-Za-z0-9_.-]+', '', s)
    if not s:
        s = "line"
    return s[:80]
