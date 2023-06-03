#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
from contextlib import contextmanager
from dataclasses import dataclass
from inspect import Signature
from typing import Callable, Dict, Iterator, List
from unittest import mock

from sematic.function import Function


@dataclass
class SematicFuncMock:
    mock: mock.MagicMock
    original: Callable
    signature: Signature


@contextmanager
def mock_sematic_funcs(
    funcs: List[Callable],
) -> Iterator[Dict[Callable, SematicFuncMock]]:
    """Mock Sematic funcs so they still return futures and check input/output types.

    To be used as a context manager:
    ```
        with mock_sematic_funcs(funcs=[pipeline_step_1, pipeline_step2]) as mocks:
            mocks[pipeline_step_1].mock.return_value = ...
            mocks[pipeline_step_2].mock.return_value = ...
            pipeline().resolve(sematic.resolvers.silent_resolver.SilentResolver())
    ```

    When a function decorated with @sematic.func is provided here, it will still
    behave as a Sematic func in that it will return a future, check input/output
    types and values during execution, and participate in helping Sematic construct
    a DAG. However, when it comes time to actually execute the code that was decorated,
    it is mocked out. This is thus useful for checking the structure of a Sematic
    pipeline or mocking out an individual Sematic func within one in case you don't want
    it to execute during testing.

    Once Sematic has an officially supported version of this, use it. See:
    https://github.com/sematic-ai/sematic/issues/221

    This code is based on:
    https://github.com/sematic-ai/sematic/blob/0176b4d94c0e013406a2e6fed085e3d54fe80394/sematic/func_testing/mock_sematic_funcs.py

    Args:
        funcs: a list of Sematic funcs you want to mock

    Yields:
        Yields a dictionary mapping the Sematic func to an object with handles to the mocks.
        For every key in the dictionary, the following is available:
        - yielded[my_func].mock  # a MagicMock that will be used when it is actually time to
          execute the @sematic.func decorated code
        - yielded[my_func].original  # the original function that was decorated with @sematic.func

    Raises:
        ValueError: if the values passed to funcs are not all @sematic.func objects
    """
    func_mocks = {}
    # trunk-ignore-begin(pylint/W0212)
    try:
        for func in funcs:
            func_mocks[func] = SematicFuncMock(
                mock=mock.MagicMock(),
                original=func._func,
                signature=func.__signature__(),
            )
            func._func = func_mocks[func].mock
            func._func.__signature__ = func_mocks[func].signature
        yield func_mocks
    finally:
        for func in funcs:
            if not isinstance(func, Function):
                raise ValueError(
                    f"mock_sematic_funcs(funcs=[...]) must be given a list of "
                    f"functions decorated with @sematic.func, but one of the "
                    f"elements was: {func}"
                )
            func._func = func_mocks[func].original
    # trunk-ignore-end(pylint/W0212)
