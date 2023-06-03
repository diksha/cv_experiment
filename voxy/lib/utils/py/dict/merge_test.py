import unittest
from typing import Dict, Iterable, Tuple

from lib.utils.py.dict.merge import (
    ConflictStrategy,
    DictionaryMerger,
    KeyConflictError,
)

DICT_MAIN = {
    "outer_key_1": {
        "inner_key_1": "value_1",
        "inner_key_2": "value_2",
        "inner_key_3": {
            "inner_inner_key_1": "value_3",
            "inner_inner_key_2": "value_4",
        },
    },
    "outer_key_2": {"inner_key_4": "value_5", "inner_key_5": "value_6"},
}


class DictionaryMergerTest(unittest.TestCase):
    def do_test_no_error(
        self,
        builder: DictionaryMerger,
        inputs: Iterable[Tuple],
        expected: Dict,
    ) -> None:
        """Helper function for testing DictionaryMerger

        Args:
            builder (DictionaryMerger): client under test
            inputs (Iterable[Tuple]): arguments for apply
            expected (Dict): expected merged output
        """
        for input_ in inputs:
            builder.apply(*input_)

        assert builder.get_merged() == expected

    def test_one_apply(self) -> None:
        builder = DictionaryMerger()
        self.do_test_no_error(
            builder,
            [
                (DICT_MAIN, "original"),
            ],
            DICT_MAIN,
        )

    def test_no_override(self) -> None:
        builder = DictionaryMerger()

        self.do_test_no_error(
            builder,
            [
                (DICT_MAIN, "original"),
                (
                    {
                        "outer_key_1": {
                            "inner_key_3": {
                                "inner_inner_key_3": "new_key",
                            },
                        },
                        "outer_key_3": 1337,
                    },
                    "addition",
                ),
            ],
            {
                "outer_key_1": {
                    "inner_key_1": "value_1",
                    "inner_key_2": "value_2",
                    "inner_key_3": {
                        "inner_inner_key_1": "value_3",
                        "inner_inner_key_2": "value_4",
                        "inner_inner_key_3": "new_key",
                    },
                },
                "outer_key_2": {
                    "inner_key_4": "value_5",
                    "inner_key_5": "value_6",
                },
                "outer_key_3": 1337,
            },
        )

    def test_override(self) -> None:
        builder = DictionaryMerger()

        self.do_test_no_error(
            builder,
            [
                (DICT_MAIN, "original"),
                (
                    {
                        "outer_key_1": {
                            "inner_key_3": {
                                "inner_inner_key_1": "overridden",
                            },
                        },
                        "outer_key_3": {
                            "inner_key_1": "new_key",
                        },
                    },
                    "addition",
                ),
            ],
            {
                "outer_key_1": {
                    "inner_key_1": "value_1",
                    "inner_key_2": "value_2",
                    "inner_key_3": {
                        "inner_inner_key_1": "overridden",
                        "inner_inner_key_2": "value_4",
                    },
                },
                "outer_key_2": {
                    "inner_key_4": "value_5",
                    "inner_key_5": "value_6",
                },
                "outer_key_3": {"inner_key_1": "new_key"},
            },
        )

    def test_override_no_update_strategy(self) -> None:
        builder = DictionaryMerger()

        self.do_test_no_error(
            builder,
            [
                (DICT_MAIN, "original"),
                (
                    {
                        "outer_key_1": {
                            "inner_key_3": {
                                "inner_inner_key_1": "overridden",
                            },
                        },
                        "outer_key_3": {
                            "inner_key_1": "new_key",
                        },
                    },
                    "addition",
                    ConflictStrategy.DO_NOT_UPDATE,
                ),
            ],
            {
                "outer_key_1": {
                    "inner_key_1": "value_1",
                    "inner_key_2": "value_2",
                    "inner_key_3": {
                        "inner_inner_key_1": "value_3",
                        "inner_inner_key_2": "value_4",
                    },
                },
                "outer_key_2": {
                    "inner_key_4": "value_5",
                    "inner_key_5": "value_6",
                },
                "outer_key_3": {
                    "inner_key_1": "new_key",
                },
            },
        )

    def test_override_throw(self) -> None:
        with self.assertRaises(KeyConflictError):
            builder = DictionaryMerger()
            builder.apply(DICT_MAIN, "original")
            builder.apply(
                {
                    "outer_key_1": {
                        "inner_key_3": {
                            "inner_inner_key_1": "already_defined",
                        }
                    }
                },
                "addition",
                ConflictStrategy.THROW_ERROR,
            )

    def test_sources(self) -> None:
        builder = DictionaryMerger()

        expected_final_sources_dict = {
            "outer_key_1.inner_key_1": ("main", ["main"]),
            "outer_key_1.inner_key_2": ("main", ["main"]),
            "outer_key_1.inner_key_3.inner_inner_key_1": (
                "conflict",
                ["main", "conflict"],
            ),
            "outer_key_1.inner_key_3.inner_inner_key_2": ("main", ["main"]),
            "outer_key_1.inner_key_4": ("no conflict", ["no conflict"]),
            "outer_key_2.inner_key_4": ("main", ["main"]),
            "outer_key_2.inner_key_5": ("main", ["main"]),
            "outer_key_3.inner_key_1": ("no conflict", ["no conflict"]),
            "outer_key_4.inner_key_1": ("conflict", ["conflict"]),
        }

        builder.apply(DICT_MAIN, "main")

        cur_keys = [
            "outer_key_1.inner_key_1",
            "outer_key_1.inner_key_2",
            "outer_key_1.inner_key_3.inner_inner_key_1",
            "outer_key_1.inner_key_3.inner_inner_key_2",
            "outer_key_2.inner_key_4",
            "outer_key_2.inner_key_5",
        ]

        for key in cur_keys:
            definition_src = builder.get_definition_source(key)
            # assert source is the same as the first source in expected dict
            assert definition_src == expected_final_sources_dict[key][1][0]
            assert builder.get_definition_sources(key) == [definition_src]

        builder.apply(
            {
                "outer_key_1": {
                    "inner_key_4": "value_4",
                },
                "outer_key_3": {
                    "inner_key_1": "value_1",
                },
            },
            "no conflict",
        )

        cur_keys += [
            "outer_key_1.inner_key_4",
            "outer_key_3.inner_key_1",
        ]

        for key in cur_keys:
            definition_src = builder.get_definition_source(key)
            assert definition_src == expected_final_sources_dict[key][1][0]
            assert builder.get_definition_sources(key) == [definition_src]

        builder.apply(
            {
                "outer_key_1": {
                    "inner_key_3": {
                        "inner_inner_key_1": "overridden",
                    }
                },
                "outer_key_4": {
                    "inner_key_1": "value_1",
                },
            },
            "conflict",
        )

        cur_keys += [
            "outer_key_4.inner_key_1",
        ]

        for key in cur_keys:
            assert (
                builder.get_definition_source(key)
                == expected_final_sources_dict[key][0]
            )
            assert (
                builder.get_definition_sources(key)
                == expected_final_sources_dict[key][1]
            )

        assert (
            builder.get_definition_sources_dict()
            == expected_final_sources_dict
        )

    def test_bad_key_delimiter(self) -> None:
        builder = DictionaryMerger(key_delimiter="_")
        with self.assertRaises(ValueError):
            builder.apply(DICT_MAIN, "original")

    def test_multi_character_key_delimiter(self) -> None:
        builder = DictionaryMerger(
            key_delimiter="please_dont_use_a_delimiter_like_this"
        )

        self.do_test_no_error(
            builder,
            [
                (DICT_MAIN, "original"),
                (
                    {
                        "outer_key_1": {
                            "inner_key_3": {
                                "inner_inner_key_1": "overridden",
                            },
                        },
                        "outer_key_3": {
                            "inner_key_1": "new_key",
                        },
                    },
                    "addition",
                ),
            ],
            {
                "outer_key_1": {
                    "inner_key_1": "value_1",
                    "inner_key_2": "value_2",
                    "inner_key_3": {
                        "inner_inner_key_1": "overridden",
                        "inner_inner_key_2": "value_4",
                    },
                },
                "outer_key_2": {
                    "inner_key_4": "value_5",
                    "inner_key_5": "value_6",
                },
                "outer_key_3": {"inner_key_1": "new_key"},
            },
        )
