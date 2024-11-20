# SPDX-FileCopyrightText: 2024 Sebastian Kempf <sebastian.kempf@uni-wuerzburg.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
from typing import Dict, Optional

from kieta_data_objs import Document
from kieta_data_objs.util import img_to_base64
from PIL import Image

from .. import Module


class ExportModule(Module):
    _MODULE_TYPE = "ExportModule"

    def __init__(
        self, stage: int, parameters: Dict | None = None, debug_mode: bool = False
    ) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.output_dir = parameters.get("output_dir", "output")
        self.format = parameters.get("format", "png")
        self.write_to_file = parameters.get("write_to_file", True)
        self.prefix = parameters.get("prefix", "")

        if self.write_to_file and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def execute(self, inpt: Document) -> Document:
        raise NotImplementedError("ExportModule is abstract and cannot be executed")


class ExportDrawnFiguresModule(ExportModule):
    _MODULE_TYPE = "ExportDrawnFiguresModule"

    """
    Exports drawn figures
    """

    def __init__(
        self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = False
    ) -> None:
        if "format" not in parameters:
            parameters["format"] = "json"
        super().__init__(stage, parameters, debug_mode)
        self.apply_to = parameters.get("apply_to", "DrawnFigure")
        self.export_references = parameters.get("export_references", True)
        self.export_overlapping_text = parameters.get("export_overlapping_text", True)
        self.filter_smaller_areas = parameters.get("filter_smaller_areas", 150)

    def execute(self, inpt: Document) -> Document:
        """
        Export in format
        [
            {
                "id": "drawn_figure_0",
                "page": 0,
                "boundingBox": [0, 0, 100, 100],
                "img": "base64",
                "objs": [
                    {
                        "id": "drawn_line_0",
                        "type": "DrawnLine",
                        "boundingBox": [0, 0, 100, 100],
                        "pts": [[0, 0], [100, 100]]
                    }
                ]
            }
        ]
        """
        export = []

        for p in inpt.pages:
            areas_on_page = list(
                sorted(
                    [
                        x
                        for x in inpt.get_area_type(self.apply_to, p.oid)
                        if x.boundingBox.area() > self.filter_smaller_areas
                    ],
                    key=lambda x: x.oid,
                )
            )

            imgs = inpt.get_img_snippets(areas_on_page, page=p.oid)

            for area, img in zip(areas_on_page, imgs):
                objs = []
                try:
                    for ref in inpt.references[area.oid]:
                        ref = inpt.get_area_obj(ref)
                        # bb = ref.boundingBox.get_in_xml_space(p.factor_width, p.factor_height)
                        bb = ref.boundingBox.get_in_img_space(
                            p.factor_width, p.factor_height
                        )
                        # strokes
                        add_info = ref.data

                        for r in inpt.references.byId[ref.oid]:
                            r = inpt.get_area_obj(r)
                            # bb = ref.boundingBox.get_in_xml_space(p.factor_width, p.factor_height)
                            bb = ref.boundingBox.get_in_img_space(
                                p.factor_width, p.factor_height
                            )
                            pts = r.data.get("pts", [])
                            if not pts:
                                try:
                                    pts = bb.tuple()
                                except Exception:
                                    continue
                            else:
                                pts = [
                                    [x[0] * p.factor_width, x[1] * p.factor_height]
                                    for x in pts
                                ]

                            objs.append(
                                {
                                    "id": r.oid,
                                    "type": r.category,
                                    "boundingBox": [bb.x1, bb.y1, bb.x2, bb.y2],
                                    "pts": pts,
                                    "draw": add_info,
                                }
                            )
                except (KeyError, TypeError) as e:
                    continue
                if self.export_overlapping_text:
                    # assume that everything is in xml space
                    for text in inpt.get_area_type("String", page=p):
                        if not text.boundingBox.overlap(area.boundingBox):
                            continue
                        else:
                            # text_bb = text.boundingBox.get_in_xml_space(p.factor_width, p.factor_height)
                            text_bb = text.boundingBox.get_in_img_space(
                                p.factor_width, p.factor_height
                            )
                            objs.append(
                                {
                                    "id": text.oid,
                                    "type": text.category,
                                    "boundingBox": [
                                        text_bb.x1,
                                        text_bb.y1,
                                        text_bb.x2,
                                        text_bb.y2,
                                    ],
                                    "content": text.data.get("content", ""),
                                }
                            )
                # find closest caption
                closest = None
                captions = list(inpt.get_area_type("Caption", page=p))
                if len(captions) == 1:
                    closest = captions[0].data.get("content", "")
                else:
                    closest_dist = float("inf")
                    for c in captions:
                        caption_bb = c.boundingBox.get_in_img_space(
                            p.factor_width, p.factor_height
                        )
                        area_bb = area.boundingBox.get_in_img_space(
                            p.factor_width, p.factor_height
                        )
                        if caption_bb.y1 < area_bb.y1:
                            continue
                        dist = caption_bb.distance(area_bb)
                        if dist < closest_dist:
                            closest = c.data.get("content", "")
                            closest_dist = dist
                try:
                    img = img_to_base64(Image.fromarray(img))
                except ValueError:
                    img = ""

                img_space_bb = area.boundingBox.get_in_img_space(
                    p.factor_width, p.factor_height
                )

                export.append(
                    {
                        "id": area.oid,
                        "page": p.oid,
                        "boundingBox": [
                            img_space_bb.x1,
                            img_space_bb.y1,
                            img_space_bb.x2,
                            img_space_bb.y2,
                        ],
                        "caption": closest,
                        "img": img,
                        "objs": objs,
                    }
                )
        return export
        return inpt
