import typing

from .elementid import ElementId, ElementType
from .mkv_element_ids import register as register_mkv_element_ids

# registers the mkv element ids that are generated by gen_element_ids.py
register_mkv_element_ids()


class MKVReadError(RuntimeError):
    pass


class ElementHeader:
    def __init__(self, eid: ElementId, data_size: typing.Optional[int]):
        self.id = eid
        self.data_size = data_size

    def has_unknown_size(self) -> bool:
        return self.data_size is None

    @classmethod
    def parse(
        cls, data: bytes
    ) -> typing.Tuple[int, typing.Optional["ElementHeader"]]:
        idn, idval = _parse_vint(data[0:4], 8)
        if idval is None:
            return 0, None

        eid = _vint_to_element_id(idn, idval)
        sizen, size = _parse_vint(data[idn : idn + 8], 8)
        if size is None:
            return 0, None

        # unknown size elements always contain other elements so
        # just skip them. we ignore them for the purposes of identifying
        # what element we are in since they can only be Segment/Cluster
        if _is_unknown_size(sizen, size):
            size = None

        return idn + sizen, cls(eid, size)

    def __str__(self) -> str:
        return f"{self.id}(len={self.data_size})"


class MKVTagParser:
    def __init__(
        self,
    ):
        self._tags: dict[str, str] = {}
        self._data: bytearray = bytearray()

    def parse(self, data: bytes) -> None:
        self._data += bytearray(data)
        while self._parse_next():
            pass

    def _parse_next(self) -> bool:
        if len(self._data) == 0:
            return False

        # try to read the header, a failure here is probably unrecoverable
        n, header = ElementHeader.parse(self._data)
        if header is None and len(self._data) > 16:
            raise MKVReadError("invalid ebml data found")

        # just ignore unknown size headers
        if header.has_unknown_size():
            self._data = self._data[n:]
            return True

        # if we have a full element, parse the whole thing
        if len(self._data) >= n + header.data_size:
            element_data = self._data[n : n + header.data_size]

            if header.id.name == "SimpleTag":
                self._parse_tag(element_data)
            elif header.id.type == ElementType.MASTER:
                self._parse_element(element_data)

            self._data = self._data[n + header.data_size :]
            return True

        return False

    def _parse_element(self, data: bytes) -> None:
        while len(data) > 0:
            n, header = ElementHeader.parse(data)
            if header is None:
                raise MKVReadError(
                    "invalid mkv data encountered while reading header"
                )

            if header.has_unknown_size():
                raise MKVReadError(
                    "unknown size element found inside a known size element"
                )

            if header.id.name == "SimpleTag":
                # this is what we are here for
                self._parse_tag(data[n : n + header.data_size])
            elif header.id.type == ElementType.MASTER:
                # recurse into master tags
                self._parse_element(data[n : n + header.data_size])

            # advance data to the next element
            data = data[n + header.data_size :]

    def _parse_tag(self, data: bytes):
        # we are assuming this is the body of a simpletag
        tagName: str = str()
        tagString: str = str()
        while len(data) > 0:
            n, header = ElementHeader.parse(data)
            if header is None:
                raise MKVReadError(
                    "invalid mkv data encountered while reading tag element header"
                )

            if header.has_unknown_size():
                raise MKVReadError(
                    "invalid mkv data: unknown size element inside known size element"
                )

            if header.id.name == "TagName":
                tagName = (
                    data[n : n + header.data_size].decode().rstrip("\x00")
                )
            elif header.id.name == "TagString":
                tagString = (
                    data[n : n + header.data_size].decode().rstrip("\x00")
                )

            data = data[n + header.data_size :]

        if len(tagName) > 0:
            self._tags[tagName] = tagString

    def tags(self) -> typing.Dict[str, str]:
        return self._tags


class MKVTagReader:
    def __init__(self, reader):
        self._reader = reader
        self._tagr = MKVTagParser()

    def read(self, size=-1) -> bytes:
        buf = self._reader.read(size)
        self._tagr.parse(buf)
        return buf

    def tags(self) -> typing.Dict[str, str]:
        return self._tagr.tags()

    def __getattr__(self, attr):
        return getattr(self._reader, attr)


def _parse_vint(data: bytes, maxsize: int) -> typing.Tuple[int, int]:
    if maxsize > 8:
        raise MKVReadError("vint maxsize > 8 unsupported")

    if len(data) < 1:
        raise MKVReadError("cannot parse empty vint data")

    mask = 1 << 7
    size = 1
    octet = data[0]

    while mask & octet == 0:
        size += 1
        mask >>= 1
        if size > maxsize:
            raise MKVReadError("invalid vint size bits")

    if len(data) < size:
        raise MKVReadError(
            f"invalid vint data, expected {size} bytes but received {len(data)}"
        )

    value = int(octet & (mask - 1))
    for octet in data[1:size]:
        value = (value << 8) + int(octet)

    return size, value


def _vint_to_element_id(width: int, val: int) -> ElementId:
    return ElementId.by_id(
        val | (((1 << 7) >> (width - 1)) << (8 * (width - 1)))
    )


def _is_unknown_size(width: int, val: int) -> bool:
    maxval = int(((1 << 7) >> (width - 1)) - 1)
    for _ in range(width - 1):
        maxval = (maxval << 8) + 0xFF
    return val == maxval
