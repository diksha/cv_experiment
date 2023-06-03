import argparse
import sys
import typing

import defusedxml.ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="generate a python source file with ebml element id data parsed from an ebml element id schema"
    )
    parser.add_argument("input", nargs="?", default="-")
    return parser.parse_args()


ELEMENT_TYPES: dict[str, str] = {
    "master": "ElementType.MASTER",
    "uinteger": "ElementType.UINTEGER",
    "binary": "ElementType.BINARY",
    "utf-8": "ElementType.UTF8",
    "float": "ElementType.FLOAT",
    "date": "ElementType.DATE",
    "integer": "ElementType.INTEGER",
    "string": "ElementType.STRING",
}


def convert_element_type(name: str) -> str:
    if name in ELEMENT_TYPES:
        return ELEMENT_TYPES[name]

    raise RuntimeError(f"Encountered unknown element type {name}")


def generate_from_file(inf: typing.TextIO) -> None:
    tree = ET.parse(inf)
    root = tree.getroot()

    print("from .elementid import ElementId, ElementType")
    print("")
    print("")
    print("def register():")
    print("    ElementId.register([")
    for element in root:
        idhex = element.attrib["id"]
        typ = convert_element_type(element.attrib["type"])
        name = element.attrib["name"]
        print(f'        ElementId({idhex}, {typ}, "{name}"),')

    print("    ])")
    print("")


def main():
    args = parse_args()

    if args.input == "-":
        generate_from_file(sys.stdin)
    else:
        with open(args.input, "r", encoding="utf-8") as inf:
            generate_from_file(inf)


if __name__ == "__main__":
    sys.exit(main())
