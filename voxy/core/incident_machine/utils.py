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

import importlib
import inspect

from core.incident_machine import machines
from core.incident_machine.machines.base import BaseStateMachine


def is_machine(cls):
    """Is subclass of base class, but not the actual base class."""
    if not inspect.isclass(cls):
        return False
    return issubclass(cls, BaseStateMachine) and cls != BaseStateMachine


def iter_machines(machines_requested):
    """Iterates through machine classes defined in machine directory."""

    if not machines_requested:
        return

    all_selected = "all" in machines_requested
    for module_name in machines.__all__:
        relative_module_name = ".{}".format(module_name)
        module = importlib.import_module(
            relative_module_name, "core.incident_machine.machines"
        )
        for _, machine in inspect.getmembers(module, is_machine):
            if all_selected or machine.NAME in machines_requested:
                yield machine
