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

from typing import List

from loguru import logger

from core.incident_machine.utils import iter_machines


class IncidentMachineController:
    def __init__(
        self,
        state_machine_monitors_requested: List[str],
        state_machine_params: dict,
        camera_uuid: str,
        dry_run: bool,
        run_uuid: str,
    ):

        if not state_machine_monitors_requested:
            state_machine_monitors_requested = []

        self._camera_uuid = camera_uuid
        self._dry_run = dry_run
        self._run_uuid = run_uuid
        self._machines_requested = state_machine_monitors_requested
        self._params = state_machine_params
        self._initialize_incident_machines()

    def process(self, state_event):
        return self._run_machines(state_event)

    def finalize(self, state_event):
        return self._run_machines(state_event)

    def _run_machines(self, state_event):
        incidents = []
        if not state_event:
            return incidents

        # Run through all machines
        for machine in self._machines:
            generated_incidents = machine.process_state_event(state_event)
            if generated_incidents is not None:
                for incident in generated_incidents:
                    incident.run_uuid = self._run_uuid
                incidents.extend(generated_incidents)

        return incidents

    def _initialize_incident_machines(self):
        self._machines = []

        for machine in iter_machines(
            machines_requested=self._machines_requested,
        ):
            self._machines.append(
                machine(self._camera_uuid, self._params.get(machine.NAME, {}))
            )
            logger.info(f"Initialized {machine.NAME} machine")
