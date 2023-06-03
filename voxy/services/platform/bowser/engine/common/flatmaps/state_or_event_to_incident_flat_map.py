import yaml
from pyflink.datastream import FlatMapFunction
from rules_python.python.runfiles import runfiles

from core.execution.nodes.incident_machine import IncidentMachineNode


class StateOrEventToIncidentFlatMap(FlatMapFunction):
    def __init__(self):
        self._incident_machine = {}

    def _load_incident_machine(self, camera_uuid):
        """Attempts to load up the relevant incident machine for a camera

        Args:
            camera_uuid (str): camera uuid to load config for
        """
        if self._incident_machine.get(camera_uuid, None) is not None:
            return

        runf = runfiles.Create()
        with open(
            runf.Rlocation(f"voxel/configs/cameras/{camera_uuid}.yaml"),
            mode="r",
            encoding="utf-8",
        ) as gcf:
            graph_config = yaml.safe_load(gcf)

        self._incident_machine[camera_uuid] = IncidentMachineNode(graph_config)

    def flat_map(self, state_or_event):
        """Consumes state/event messages and runs them through an incident machine

        Args:
            state_or_event (Union[StatePb, EventPb]): _description_

        Yields:
            _type_: _description_
        """

        self._load_incident_machine(state_or_event.camera_uuid)

        if (
            self._incident_machine.get(state_or_event.camera_uuid, None)
            is not None
        ):
            incidents = self._incident_machine.get(
                state_or_event.camera_uuid
            ).process([state_or_event])

            for inc in incidents:
                yield inc
