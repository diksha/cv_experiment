#
# Copyright 2020-2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import unittest

from core.common.utils.recursive_namespace import RecursiveSimpleNamespace


class NamespaceTest(unittest.TestCase):
    """
    Tests RecursiveSimpleNamespace
    """

    def test_recursive_dictionary(self) -> None:
        """
        Tests by creating dictionary and generating recursive simple namespace
        """
        items = {
            "foo": 1,
            "bar": 2,
            "foobar": {
                "nested_type": "value",
                "nested_list": ["value1", {"a": 1, "b": 2, "c": 3}],
            },
        }
        namespace = RecursiveSimpleNamespace(**items)
        self.assertEqual(namespace.foo, 1)
        self.assertEqual(namespace.bar, 2)
        print(namespace.foobar)
        self.assertEqual(namespace.foobar.nested_type, "value")
        self.assertEqual(namespace.foobar.nested_list[0], "value1")
        self.assertEqual(namespace.foobar.nested_list[1].a, 1)
        self.assertEqual(namespace.foobar.nested_list[1].b, 2)
        self.assertEqual(namespace.foobar.nested_list[1].c, 3)


if __name__ == "__main__":
    unittest.main()
