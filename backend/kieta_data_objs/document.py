# SPDX-FileCopyrightText: 2024 Sebastian Kempf <sebastian.kempf@uni-wuerzburg.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime
import json
import logging
import uuid
from typing import Any, Callable, Dict, Generator, Iterable, List, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from PIL import Image

from .area import Area
from .base import DocObj
from .font import Font
from .geometry import BoundingBox
from .link import Link
from .normalizedObject import NormalizedObj
from .ontologicalEntity import Entity
from .page import Page
from .revision import Revision
from .util import base64_to_img, img_to_base64

logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)


class Document(DocObj):
    def __init__(
        self,
        oid: str,
        pages: NormalizedObj[Page] = None,
        areas: NormalizedObj[Area] = None,
        links: NormalizedObj[Link] = None,
        references: NormalizedObj[List[str]] = None,
        revisions: List[Revision] = None,
        fonts: List[Font] = None,
        onto_information: List[Entity] = None,
        metadata: Dict[str, Any] = None,
        raw_pdf: bytes = None,
    ):
        self.oid: str = oid
        self.pages: NormalizedObj[Page] = pages if pages else NormalizedObj()
        self.areas: NormalizedObj[Area] = areas if areas else NormalizedObj()
        self.fonts: List[Font] = fonts if fonts else list()
        self.links: NormalizedObj[Link] = links if links else NormalizedObj()
        self.references: NormalizedObj[List[str]] = (
            references if references else NormalizedObj()
        )
        self.revisions: List[Revision] = (
            revisions
            if revisions
            else [
                Revision(
                    datetime.datetime.now().isoformat(timespec="milliseconds"),
                    set(),
                    comment="Initial Revision",
                )
            ]
        )
        self.onto_information: List[Entity] = (
            onto_information if onto_information else list()
        )
        self.metadata: Dict[str, Any] = metadata if metadata else dict()
        self.raw_pdf: bytes = raw_pdf

    def add_area(
        self,
        page: Union[str, Page],
        category: str,
        boundingBox: BoundingBox,
        area_id: str = None,
        data: Any = None,
        referenced_by: List[str] = None,
        references: List[str] = None,
        confidence: float = None,
        convert_to_xml: bool = False,
        id_prefix: str = "",
    ) -> str:
        if isinstance(page, Page):
            page = page.oid

        if page not in self.pages.allIds:
            raise ValueError(f"Page '{page}' does not exist.")

        while not area_id or area_id in self.areas.allIds:
            area_id = f"{self.oid}-{self.pages[page].number}-{category}-{id_prefix}-{str(uuid.uuid4())[:8]}"

        if convert_to_xml:
            boundingBox = boundingBox.get_in_xml_space(
                self.pages[page].factor_width, self.pages[page].factor_height
            )

        ar = Area(
            area_id,
            boundingBox=boundingBox,
            category=category,
            data=data if data else {},
            confidence=confidence,
        )
        self.areas.append(ar)
        self.references[page].append(area_id)
        self.revisions[-1].objects.add(area_id)

        if referenced_by:
            if isinstance(referenced_by, str):
                self.references.byId.setdefault(referenced_by, []).append(area_id)
            else:
                for r in referenced_by:
                    self.references.byId.setdefault(r, []).append(area_id)

        if references:
            for r in references:
                self.references.byId.setdefault(area_id, []).append(r)

        return area_id

    def cleanup_references(self):
        self.references.byId = {
            k: [x for x in v if x in self.areas.allIds]
            for k, v in self.references.byId.items()
        }
        self.references.byId = {k: v for k, v in self.references.byId.items() if v}

    def delete_areas(self, area_ids: List[str]):
        area_ids_set = set(area_ids)
        for area_id in area_ids:
            self.areas.remove(area_id)
            self.references.remove(area_id)

        for k, v in self.references.byId.items():
            self.references.byId[k] = [x for x in v if x not in area_ids_set]

        self.revisions[-1].del_objs.update(area_ids)

    def delete_area(self, area_id: str):
        self.areas.remove(area_id)
        self.references.remove(area_id)

        for ref in self.references.byId.values():
            if area_id in ref:
                ref.remove(area_id)

        self.revisions[-1].del_objs.add(area_id)

    def replace_area(self, old_id: str, new_area: Area) -> None:
        assert self.areas.byId[old_id].category == new_area.category
        self.areas.byId[old_id] = new_area

        for ref in self.references.byId.values():
            if old_id in ref:
                ref[ref.index(old_id)] = new_area.oid

        self.revisions[-1].del_objs.add(old_id)

    def replace_area_multiple(self, old_id: str, new_areas: List[Area]) -> None:
        page = self.find_page_of_area(old_id)
        self.replace_area(old_id, new_areas[0])
        for a in new_areas[1:]:
            self.add_area(
                page, a.category, a.boundingBox, data=a.data, confidence=a.confidence
            )

    def find_page_of_area(self, area: Union[str, Area]) -> str:
        if isinstance(area, Area):
            area = area.oid
        for page in self.pages.allIds:
            if area in self.references.byId[page]:
                return page
        raise ValueError(f"Area '{area}' not found in any page references.")

    def get_latest_areas(self) -> List[str]:
        l = set()
        for rev in self.revisions:
            l.update(rev.adjust_objs(l))
        return list(l)

    def _get_adjacent_areas(
        self, area_id: Union[str, Area], category: str, direction: str
    ) -> List[Area]:
        if isinstance(area_id, str):
            area_id = self.areas.byId[area_id]
        page = self.find_page_of_area(area_id)
        if not page:
            raise ValueError(f"Area '{area_id}' not found on any page.")

        bbox = area_id.boundingBox
        page_data = self.pages.byId[page]
        x_shift, y_shift = 0, 0

        if direction == "left" and bbox.x1 > 0:
            x_shift = -1
        elif direction == "right" and bbox.x2 < page_data.xml_width:
            x_shift = 1
        elif direction == "above" and bbox.y1 > 0:
            y_shift = -1
        elif direction == "below" and bbox.y2 < page_data.xml_height:
            y_shift = 1
        else:
            return []

        return self.get_areas_at_position(
            page, bbox.x1 + x_shift, bbox.y1 + y_shift, category
        )

    def get_areas_left(self, area_id: Union[str, Area], category: str) -> List[Area]:
        return self._get_adjacent_areas(area_id, category, "left")

    def get_areas_right(self, area_id: Union[str, Area], category: str) -> List[Area]:
        return self._get_adjacent_areas(area_id, category, "right")

    def get_areas_above(self, area_id: Union[str, Area], category: str) -> List[Area]:
        return self._get_adjacent_areas(area_id, category, "above")

    def get_areas_below(self, area_id: Union[str, Area], category: str) -> List[Area]:
        return self._get_adjacent_areas(area_id, category, "below")

    def get_areas_at_position(
        self, page: str, x: int, y: int, category: str
    ) -> List[Area]:
        return [
            area
            for area_id in self.references.byId[page]
            if (area := self.areas.byId[area_id]).category == category
            and area.boundingBox.x1 <= x <= area.boundingBox.x2
            and area.boundingBox.y1 <= y <= area.boundingBox.y2
        ]

    def get_area_obj(self, area_id: str) -> Union[Area, None]:
        return self.areas.byId.get(area_id, None)

    def get_area_data_value(
        self, area: Union[str, Area], val: str = "content"
    ) -> List[Any]:
        if not isinstance(area, Area):
            area = self.get_area_obj(area)
        if not area:
            return []
        g = area.data.get(val, None)
        if not g:
            refs = self.references.byId.get(area.oid, None)
            if refs:
                return [
                    self.get_area_obj(ref).data.get(val, None)
                    for ref in refs
                    if self.get_area_obj(ref).data.get(val, None)
                ]
        return [g] if g else []

    def add_link(
        self,
        category: str,
        frm: str,
        to: str,
        directed: bool = False,
        link_id: str = None,
        association: str = None,
        page: str = "",
    ):
        if not link_id:
            link_id = f"{self.pages.byId.get(page, {'number': page}).get('number', page)}-{category}-{str(uuid.uuid4())[:3]}"
        li = Link(category, frm, to, directed, link_id)
        self.links.append(li)
        self.references.byId[page].append(link_id)
        self.revisions[-1].objects.add(link_id)
        if association:
            self.references.byId.setdefault(association, []).append(link_id)
        return link_id

    def add_revision(self, name: str):
        self.revisions.append(
            Revision(
                datetime.datetime.now().isoformat(timespec="milliseconds"),
                set(),
                comment=name,
            )
        )

    def add_page(self, page: Page = None, img: str = None):
        if not page:
            if isinstance(img, str):
                open_img = Image.open(img)
            elif isinstance(img, np.ndarray):
                open_img = Image.fromarray(img)
            else:
                raise Exception("Invalid image type")
            page = Page(
                f"page-{str(len(self.pages.allIds))}",
                len(self.pages.allIds),
                img_to_base64(open_img),
                img_width=open_img.width,
                img_height=open_img.height,
                xml_width=open_img.width,
                xml_height=open_img.height,
            )
        self.pages.byId[page.oid] = page
        self.references.byId[page.oid] = []
        return page.oid

    def get_area_type(
        self, category: str, page: Union[str, int] = ""
    ) -> Iterable[Area]:
        return self.get_area_by(lambda x: x.category == category, page)

    def get_area_by(
        self, fun: Callable[[Area], bool], page: Union[str, int] = ""
    ) -> Iterable[Area]:
        p = ""
        if isinstance(page, Page):
            p = page.oid
        elif page in self.pages.allIds:
            p = page
        elif f"page-{page}" in self.pages.allIds:
            p = f"page-{page}"

        if page and not p:
            return []
        elif not page:
            return (a for a in self.areas.byId.values() if fun(a))
        else:
            return (
                self.areas.byId[a]
                for a in self.references.byId[p]
                if fun(self.areas.byId[a])
            )

    def get_img_snippets(
        self, areas: List[Area], padding: Tuple[int, int] = (0, 0), page: str = None
    ) -> Generator[ArrayLike, None, None]:
        p = self.find_page_of_area(areas[0]) if not page else page
        img = np.array(base64_to_img(self.pages.byId[p].img))

        for area in areas:
            bb = area.boundingBox.get_in_img_space(
                self.pages.byId[p].factor_width, self.pages.byId[p].factor_height
            )
            yield img[
                int(bb.y1) - padding[1] : int(bb.y2) + padding[1],
                int(bb.x1) - padding[0] : int(bb.x2) + padding[0],
                :,
            ]

    def get_img_snippet(
        self, area_id: str, as_string: bool = True, padding: Tuple[int, int] = (0, 0)
    ) -> Union[str, Image.Image]:
        area = self.get_area_obj(area_id) if not isinstance(area_id, Area) else area_id
        if not area:
            raise Exception(f"Area {area_id} does not exist")
        p = self.find_page_of_area(area)
        return self.get_img_snippet_from_bb(area.boundingBox, p, as_string, padding)

    def get_img_snippet_from_bb(
        self,
        bb: BoundingBox,
        p: str,
        as_string: bool,
        padding: Tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        if bb:
            bb = bb.get_in_img_space(
                self.pages.byId[p].factor_width, self.pages.byId[p].factor_height
            )
            img = (
                base64_to_img(self.pages.byId[p].img)
                if isinstance(self.pages.byId[p].img, str)
                else self.pages.byId[p].img
            )
            cropped = Image.fromarray(
                np.array(img)[
                    max(int(bb.y1) - padding[1], 0) : min(
                        int(bb.y2) + padding[1], img.size[1] - 1
                    ),
                    max(int(bb.x1) - padding[0], 0) : min(
                        int(bb.x2) + padding[0], img.size[0] - 1
                    ),
                    :,
                ]
            )
            return cropped if not as_string else img_to_base64(cropped)
        raise Exception(f"Invalid bounding box")

    def get_img_page(
        self, page: str, as_string: bool = True
    ) -> Union[str, Image.Image]:
        p = page if page in self.pages.allIds else ""
        return (
            (
                self.pages.byId[p].img
                if as_string
                else base64_to_img(self.pages.byId[p].img)
            )
            if p
            else ""
        )

    def transpose_page(self, page_id: str):
        for obj in self.references.byId[page_id]:
            self.areas.byId[obj].boundingBox.transpose()

    @staticmethod
    def from_dic(d: Dict) -> "Document":
        if (
            not isinstance(d, dict)
            or not d.get("oid")
            or not d.get("pages")
            or not d.get("areas")
        ):
            raise Exception(f"Invalid input: {d}")

        try:
            return Document(
                oid=d["oid"],
                pages=NormalizedObj.from_dic(d["pages"], "pages"),
                areas=NormalizedObj.from_dic(d["areas"], "areas"),
                links=NormalizedObj.from_dic(d["links"], "links")
                if d.get("links")
                else NormalizedObj(),
                references=NormalizedObj.from_dic(d["references"], "references")
                if d.get("references")
                else NormalizedObj(),
                revisions=[Revision.from_dic(x) for x in d.get("revisions", set())],
                fonts=[Font.from_dic(x) for x in d.get("fonts", [])]
                if d.get("fonts")
                else [],
                metadata=d.get("metadata", {}),
                raw_pdf=d.get("raw_pdf"),
            )
        except Exception as e:
            import traceback

            print(e, traceback.format_exc())

    def to_dic(self) -> Dict:
        return {
            "oid": self.oid,
            "pages": self.pages.to_dic(),
            "areas": self.areas.to_dic(),
            "links": self.links.to_dic(),
            "references": self.references.to_dic(),
            "revisions": [x.to_dic() for x in self.revisions],
            "fonts": [x.to_dic() for x in self.fonts],
            "onto_information": [x.to_dic() for x in self.onto_information],
            "metadata": self.metadata,
            "raw_pdf": self.raw_pdf,
        }

    def to_json(self) -> str:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                import base64

                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, datetime.datetime):
                    return obj.isoformat()
                if isinstance(obj, set):
                    return list(obj)
                if isinstance(obj, bytes):
                    try:
                        return obj.decode("utf-8")
                    except UnicodeDecodeError:
                        return base64.b64encode(obj).decode("utf-8")
                return super(NpEncoder, self).default(obj)

        return json.dumps(self.to_dic(), cls=NpEncoder)

    def is_referenced_by(self, area: Union[Area, str], type: List[str] = None) -> bool:
        if isinstance(area, Area):
            area = area.oid
        for ref in self.references.allIds:
            if ref in self.pages.allIds:
                continue
            if type and self.areas.byId[ref].category not in type:
                continue
            if area in self.references.byId[ref]:
                return True
        return False
