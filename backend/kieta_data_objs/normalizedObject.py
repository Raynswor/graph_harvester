# SPDX-FileCopyrightText: 2024 Sebastian Kempf <sebastian.kempf@uni-wuerzburg.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Dict, Generic, List, TypeVar, Union

from .area import Area
from .link import Link
from .page import Page

T = TypeVar("T", Page, Area, List[str], Link)


class NormalizedObj(Generic[T]):
    def __init__(self, byId: Dict[str, T] = None):
        self.byId: Dict[str, T] = byId if byId else dict()
        self.allIds: List[str] = self.byId.keys()

    def __getitem__(self, key: str) -> Union[T, None]:
        if isinstance(key, int):
            return self.byId.get(list(self.allIds)[key])
        else:
            return self.byId.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.byId

    def __len__(self):
        return len(self.byId)

    def __iter__(self):
        return iter(self.byId.values())

    def append(self, obj: T):
        if obj.oid in self.byId:
            raise KeyError(f"Object with id {obj.oid} already exists")
        self.byId[obj.oid] = obj
        # self.allIds.append(obj.oid)

    def remove(self, obj: T):
        try:
            if not isinstance(obj, str):
                del self.byId[obj.oid]
            else:
                del self.byId[obj]
        except KeyError:
            pass
        # self.allIds.remove(obj.oid)

    def remove_multiple(self, objs: List[T]):
        for obj in objs:
            self.remove(obj)

    @staticmethod
    def from_dic(dic, what: str) -> "NormalizedObj":
        cls_map = {"areas": Area, "pages": Page, "links": Link}
        cls = cls_map.get(what)

        def rec_obj_creation(part, cls):
            return {k: cls.from_dic(v) for k, v in part.items()} if cls else part

        byId = rec_obj_creation(dic["byId"], cls) if cls else dic["byId"]
        return NormalizedObj(byId)

    def to_dic(self) -> Dict:
        def rec_obj_to_dic(part):
            return {
                k: v.to_dic() if hasattr(v, "to_dic") else v for k, v in part.items()
            }

        return {"byId": rec_obj_to_dic(self.byId), "allIds": list(self.byId.keys())}
