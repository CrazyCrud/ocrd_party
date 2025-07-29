import os
import uuid
from functools import cached_property

import torch
from PIL import Image
from ocrd import Processor, OcrdPage, OcrdPageResult
from ocrd_models.ocrd_page import TextEquivType, CoordsType
from ocrd_utils import points_from_polygon, coordinates_for_segment

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
        print(f"ocrd-party {self.version}, party {party.__version__}")

    def setup(self):
        """
        Load the Party model once and prepare Fabric.
        """
        model_path = self.parameter['model_dir']
        # Load safetensors model (must be named model.safetensors or passed explicitly)
        self.model: PartyModel = PartyModel.from_safetensors(model_path)
        # Choose device (e.g. "cpu" or "cuda:0")
        device = self.parameter.get('device', 'cpu')
        self.model.to(device)
        # Fabric for mixed precision / device management
        self.fabric = Fabric(accelerator=device, precision="fp16" if self.parameter.get('autocast', False) else "fp32")
        self.batch_size = int(self.parameter.get('batch_size', 4))

    @cached_property
    def _prompt_mode(self):
        # PartyModel.line_prompt_mode is 'boxes' or 'curves'
        return self.model.line_prompt_mode

    def process_page_pcgts(self, *input_pcgts, page_id=None):
        pcgts = input_pcgts[0]
        page = pcgts.get_Page()
        page_image, _, _ = self.workspace.image_from_page(page, page_id)

        lines = page.get_TextLine()
        if not lines:
            self.log.warning(f"No TextLine on page {page_id}")
            return OcrdPageResult(pcgts)

        # Build baseline container
        from kraken.containers import Segmentation, BaselineLine
        blls = []
        for line in lines:
            pts = [(pt.x, pt.y) for pt in (line.get_Baseline() or line.get_Coords()).points]
            blls.append(BaselineLine(id=line.id, baseline=pts, boundary=pts, tags=[]))

        seg = Segmentation(
            text_direction=self.parameter['text_direction'],
            imagename=page_id,
            type='baselines' if self._prompt_mode == 'curves' else 'bbox',
            lines=blls, regions={}, script_detection=False, line_orders=[]
        )

        # Predict
        pred_iter = batched_pred(
            model=self.model, im=page_image, bounds=seg,
            fabric=self.fabric, batch_size=self.batch_size, add_lang_token=False
        )
        for rec in pred_iter:
            avg_conf = float(sum(rec.confidences) / len(rec.confidences))
            line_obj = next(l for l in lines if l.id == rec.line.id)
            line_obj.set_TextEquiv([TextEquivType(Unicode=rec.prediction, conf=avg_conf)])

        # Update region & page TextEquiv for consistency
        _page_update_higher_textequiv_levels("line", pcgts)

        return OcrdPageResult(pcgts)


def _page_update_higher_textequiv_levels(level, pcgts):
    """Update the TextEquivs of all higher PAGE-XML hierarchy levels for consistency.

    Starting with the hierarchy level `level`chosen for processing, join all first
    TextEquiv (by the rules governing the respective level) into TextEquiv of the next
    higher level, replacing them.
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
                                (
                                    glyph.get_TextEquiv()[0].Unicode
                                    if glyph.get_TextEquiv()
                                    else ""
                                )
                                for glyph in glyphs
                            )
                            word.set_TextEquiv(
                                [TextEquivType(Unicode=word_unicode)]
                            )  # remove old
                    line_unicode = " ".join(
                        word.get_TextEquiv()[0].Unicode if word.get_TextEquiv() else ""
                        for word in words
                    )
                    line.set_TextEquiv(
                        [TextEquivType(Unicode=line_unicode)]
                    )  # remove old
            region_unicode = "\n".join(
                line.get_TextEquiv()[0].Unicode if line.get_TextEquiv() else ""
                for line in lines
            )
            region.set_TextEquiv([TextEquivType(Unicode=region_unicode)])  # remove old
