from dataclasses import dataclass
from typing import List, Union

FilterValue = Union[bool, str, List[str]]


@dataclass
class Filter:
    key: str
    value: FilterValue
