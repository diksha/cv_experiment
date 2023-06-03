import enum
import typing
from dataclasses import dataclass

ElementType = enum.Enum(
    "ElementType",
    [
        "MASTER",
        "UINTEGER",
        "BINARY",
        "UTF8",
        "FLOAT",
        "DATE",
        "INTEGER",
        "STRING",
        "UNKNOWN",
    ],
)


@dataclass
class ElementId:
    id: int
    type: ElementType
    name: str

    _element_ids: typing.ClassVar[typing.Dict[int, "ElementId"]] = {}
    _element_names: typing.ClassVar[typing.Dict[int, "ElementId"]] = {}

    @classmethod
    def register(
        cls, id_or_ids: typing.Union["ElementId", typing.List["ElementId"]]
    ) -> None:
        if not isinstance(id_or_ids, list):
            id_or_ids = [id_or_ids]

        for eid in id_or_ids:
            cls._element_ids[eid.id] = eid
            cls._element_names[eid.name] = eid

    @classmethod
    def by_id(cls, eid: int) -> "ElementId":
        if eid in cls._element_ids:
            return cls._element_ids[eid]
        return ElementId(eid, ElementType.UNKNOWN, "UNKNOWN")

    @classmethod
    def by_name(cls, name: str) -> "ElementId":
        if name in cls._element_names:
            return cls._element_names[name]
        raise IndexError

    def __str__(self) -> str:
        return f"{self.name}<id={hex(self.id)},type={self.type}>"


# pulled from https://www.rfc-editor.org/rfc/rfc8794.html#name-element-id
ElementId.register(
    [
        ElementId(0x1A45DFA3, ElementType.MASTER, "EBML"),
        ElementId(0x4286, ElementType.UINTEGER, "EBMLVersion"),
        ElementId(0x42F7, ElementType.UINTEGER, "EBMLReadVersion"),
        ElementId(0x42F2, ElementType.UINTEGER, "EBMLMaxIDLength"),
        ElementId(0x42F3, ElementType.UINTEGER, "EBMLMaxSizeLength"),
        ElementId(0x4282, ElementType.STRING, "DocType"),
        ElementId(0x4287, ElementType.UINTEGER, "DocTypeVersion"),
        ElementId(0x4285, ElementType.UINTEGER, "DocTypeReadVersion"),
        ElementId(0x4281, ElementType.UINTEGER, "DocTypeExtension"),
        ElementId(0x4283, ElementType.STRING, "DocTypeExtensionName"),
        ElementId(0x4284, ElementType.UINTEGER, "DocTypeExtensionVersion"),
        ElementId(0xBF, ElementType.BINARY, "CRC-32"),
        ElementId(0xEC, ElementType.BINARY, "Void"),
    ]
)
