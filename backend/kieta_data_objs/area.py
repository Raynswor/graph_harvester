# SPDX-FileCopyrightText: 2024 Sebastian Kempf <sebastian.kempf@uni-wuerzburg.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Any, Dict, Optional

from .base import ObjWithID
from .geometry import BoundingBox


class Area(ObjWithID):
    def __init__(
        self,
        oid: str,
        category: str,
        boundingBox: BoundingBox,
        data: Any = None,
        confidence: float = None,
    ):
        super().__init__(oid)
        self.category: str = category
        self.boundingBox: BoundingBox = boundingBox
        self.data: Any = data if data else {}
        self.confidence: Optional[float] = confidence

    def __repr__(self) -> str:
        return f"<{self.oid}: {self.boundingBox} {self.data if self.data else ''}>"

    def merge(
        self, other: "Area", merge_data: bool = True, merge_confidence: bool = True
    ):
        self.boundingBox.expand(other.boundingBox)
        if merge_data:
            for k, v in other.data.items():
                if k in self.data:
                    if isinstance(self.data[k], list):
                        self.data[k].extend(v)
                    elif isinstance(self.data[k], dict):
                        self.data[k].update(v)
                    elif isinstance(self.data[k], str):
                        self.data[k] += v
                    else:
                        self.data[k] = v
                else:
                    self.data[k] = v
        if merge_confidence and other.confidence is not None:
            self.confidence = max(self.confidence, other.confidence)

    @staticmethod
    def from_dic(d: Dict) -> "Area":
        return Area(
            d["oid"],
            d["category"],
            BoundingBox.from_dic(d["boundingBox"]),
            data=d.get("data"),
            confidence=d.get("confidence"),
        )

    def to_dic(self) -> Dict:
        dic = {
            "oid": self.oid,
            "category": self.category,
            "boundingBox": self.boundingBox.to_dic(),
        }
        if self.data:
            dic["data"] = self.data
        if self.confidence is not None:
            dic["confidence"] = self.confidence
        return dic
