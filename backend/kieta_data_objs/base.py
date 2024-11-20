# SPDX-FileCopyrightText: 2024 Sebastian Kempf <sebastian.kempf@uni-wuerzburg.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import abc
from typing import Dict


class DocObj(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def from_dic(d: Dict):
        pass

    @abc.abstractmethod
    def to_dic(self) -> Dict:
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}"


class ObjWithID(DocObj, abc.ABC):
    def __init__(self, oid: str):
        self.oid = oid

    def __hash__(self) -> int:
        return hash(self.oid)

    def __eq__(self, other: "ObjWithID") -> bool:
        if not isinstance(other, ObjWithID):
            return False
        return getattr(self, "oid", None) == getattr(other, "oid", None)
