# trunk-ignore-all(pylint/C0413,flake8/E402)
import glob
import os
import typing as t
from dataclasses import dataclass
from distutils.util import strtobool
from functools import cached_property

import django
from django.db.models.functions import Lower
from django.db.models.query import QuerySet
from termcolor import colored

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.portal.voxel.settings")
django.setup()


import yaml
from loguru import logger
from rules_python.python.runfiles import runfiles

from core.portal.api.models.incident_type import (
    CameraIncidentType,
    IncidentType,
)
from core.portal.devices.models.camera import Camera


def _print_skipped_message() -> None:
    print(f"    - {colored('Skipped', 'yellow')}")


def _print_done_message() -> None:
    print(f"    - {colored('Done', 'green')}")


def _print_error_message(error: str) -> None:
    print(f"    - {colored(f'Error: {error}', 'red')}")


def _print_section_header(title: str) -> None:
    print("\n----------------------------------------\n")
    print(title)
    print("\n----------------------------------------")


@dataclass
class CameraConfigData:
    camera_uuid: str
    monitors_requested: t.List[str]
    state_machine_monitors_requested: t.List[str]

    @property
    def all_monitors_requested(self) -> t.List[str]:
        """Get all requested monitors.

        Returns:
            t.List[str]: all requested monitors
        """
        all_monitors = (
            self.monitors_requested + self.state_machine_monitors_requested
        )
        return list(set(all_monitors))


class CameraConfigFileReader:

    _BAZEL_WORKSPACE_NAME = "voxel"
    _GRAPH_CONFIG_PATH = "configs/cameras"
    _ALL_CAMERAS_WILDCARD = "**/**/**/*.yaml"

    def __init__(self):
        super().__init__()
        self._bazel_runfiles = runfiles.Create()
        self._config_paths = self._get_config_paths()

    def _get_config_paths(self) -> t.List[str]:
        """Get all config paths.

        Returns:
            t.List[str]: all config paths
        """
        return self._filter_config_path_list(
            glob.glob(
                os.path.join(
                    self._GRAPH_CONFIG_PATH, self._ALL_CAMERAS_WILDCARD
                )
            )
        )

    def _get_runfiles_location(self, filepath: str) -> str:
        """Get a path which can be used to reference a file from Bazel runfiles.

        Args:
            filepath (str): filepath, relative to workspace root

        Returns:
            str: runfiles file path
        """
        return self._bazel_runfiles.Rlocation(
            f"{self._BAZEL_WORKSPACE_NAME}/{filepath}"
        )

    def _filter_config_path_list(
        self, config_paths: t.List[str]
    ) -> t.List[str]:
        """Filter config path list and return new list.

        Args:
            config_paths (t.List[str]): config path list

        Returns:
            t.List[str]: filtered config path list
        """
        return [
            config_path
            for config_path in config_paths
            # Exclude calibration configs
            if not config_path.endswith("calibration.yaml")
        ]

    def _read_yaml_file_as_dict(self, filepath: str) -> t.Optional[dict]:
        """Read the specified YAML file and convert the content to a dict.

        Args:
            filepath (str): YAML file path

        Returns:
            t.Optional[dict]: YAML file data as dict
        """
        runfiles_location = self._get_runfiles_location(filepath)
        with open(runfiles_location, "r", encoding="utf-8") as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError as exc:
                logger.exception(exc)
        return None

    def _get_config_data(
        self, config_path: str
    ) -> t.Optional[CameraConfigData]:
        """Get config data from the specified config file.

        Args:
            config_path (str): config file path

        Returns:
            t.Optional[CameraConfigData]: config data
        """
        config_dict = self._read_yaml_file_as_dict(config_path)
        if not config_dict:
            logger.warning(f"Config file is empty or invalid: {config_path}")
            return None

        camera_uuid = config_dict.get("camera_uuid")
        if not camera_uuid:
            logger.warning(
                f"Config file is missing camera data: {config_path}"
            )
            return None

        incident_config = config_dict.get("incident")
        if not incident_config:
            logger.warning(
                f"Config file is missing incident config: {config_path}"
            )
            return None

        monitors_requested = incident_config.get("monitors_requested", [])
        state_machine_monitors_requested = incident_config.get(
            "state_machine_monitors_requested", []
        )
        return CameraConfigData(
            camera_uuid=camera_uuid,
            monitors_requested=monitors_requested,
            state_machine_monitors_requested=state_machine_monitors_requested,
        )

    def get_all_configs(self) -> t.List[CameraConfigData]:
        """Get data for all configs.

        Returns:
            t.List[CameraConfigData]: list of config data objects
        """
        all_configs: t.List[CameraConfigData] = []
        for config_path in self._config_paths:
            data = self._get_config_data(config_path)
            if data:
                all_configs.append(data)
        return all_configs


class SyncCameras:

    # Monitor names don't always map 1:1 with incident type
    # keys, so we have some known monitor/incident type mappings
    _monitor_to_incident_type_key_map = {
        "random_spill": "spill",
        "parking": "parking_duration",
        "no_stop_at_aisle_end": "no_stop_at_end_of_aisle",
        "open_door": "open_door_duration",
        "door_intersection": "no_stop_at_door_intersection",
        "intersection": "no_stop_at_intersection",
        "motion_detection": "production_line_down",
    }

    @property
    def _production_cameras(self) -> QuerySet[Camera]:
        return Camera.objects.exclude(
            organization__is_sandbox=True,
        )

    @cached_property
    def _camera_map(self) -> t.Dict[str, Camera]:
        """Get a map of camera UUIDs to camera objects.

        Returns:
            t.Dict[str, Camera]: camera map
        """
        return {c.uuid.lower(): c for c in self._production_cameras}

    @cached_property
    def _incident_type_map(self) -> t.Dict[str, IncidentType]:
        """Get a map of incident type keys to incident type objects.

        Returns:
            t.Dict[str, IncidentType]: incident type map
        """
        return {it.key.lower(): it for it in IncidentType.objects.all()}

    def _get_incident_type_key_for_monitor(
        self, monitor: str
    ) -> t.Optional[str]:
        """Get the incident type key associated with a monitor.

        Args:
            monitor (str): monitor name

        Returns:
            t.Optional[str]: incident type key if one exists, otherwise None
        """
        monitor = monitor.lower()

        # If a monitor matches an incident type key exactly, return the
        # monitor value
        if monitor in self._incident_type_map:
            return monitor

        # If a monitor matches a known monitor/type map, return the
        # corresponding incident type key
        if monitor in self._monitor_to_incident_type_key_map:
            return self._monitor_to_incident_type_key_map[monitor]

        logger.error(f"Unknown monitor requested: {monitor}")
        return None

    @cached_property
    def _desired_camera_incident_type_pairs(self) -> t.Set[t.Tuple[str, str]]:
        """Get camera incident type pairs which should be enabled.

        Returns:
            t.Set[t.Tuple[str, str]]: set of camera incidenet type pairs
        """
        file_reader = CameraConfigFileReader()
        all_configs = file_reader.get_all_configs()

        pairs = set()
        for config in all_configs:
            for monitor in config.all_monitors_requested:
                incident_type_key = self._get_incident_type_key_for_monitor(
                    monitor
                )
                if incident_type_key:
                    pairs.add((config.camera_uuid, incident_type_key))
        return pairs

    @cached_property
    def _current_camera_incident_type_pairs(self) -> t.Set[t.Tuple[str, str]]:
        """Get camera incident type pairs which are currently enabled.

        Returns:
            t.Set[t.Tuple[str, str]]: set of camera incident type pairs
        """
        camera_incident_type_pairs = (
            CameraIncidentType.objects.filter(
                camera__id__in=self._production_cameras
            )
            .annotate(
                camera_uuid=Lower("camera__uuid"),
                incident_type_key=Lower("incident_type__key"),
            )
            .values_list("camera_uuid", "incident_type_key")
        )
        return set(camera_incident_type_pairs)

    def _enable_camera_incident_type(
        self,
        camera_uuid: str,
        incident_type_key: str,
    ) -> None:
        """Enable a camera incident type.

        Args:
            camera_uuid (str): camera UUID
            incident_type_key (str): incident type key
        """
        errors: t.List[str] = []
        camera = self._camera_map.get(camera_uuid)
        incident_type = self._incident_type_map.get(incident_type_key)

        if not camera:
            errors.append("camera doesn't exist in portal DB")
        if not incident_type:
            errors.append("incident type doesn't exist in portal DB")

        label = f"{incident_type_key} @ {camera_uuid}"

        if errors:
            print(f"\n{label}")
            for error in errors:
                _print_error_message(error)
        else:
            prompt = f"\n{label} | enable? (y/n): "
            if strtobool(input(prompt)):
                CameraIncidentType.objects.update_or_create(
                    camera=camera,
                    incident_type=incident_type,
                    defaults={"enabled": True},
                )
                _print_done_message()
            else:
                _print_skipped_message()

    def _disable_camera_incident_type(
        self,
        camera_uuid: str,
        incident_type_key: str,
    ) -> None:
        """Disable a camera incident type.

        Args:
            camera_uuid (str): camera UUID
            incident_type_key (str): incident type key
        """
        label = f"{incident_type_key} @ {camera_uuid}"
        prompt = f"\n{label} | disable? (y/n): "
        if strtobool(input(prompt)):
            camera_incident_type = CameraIncidentType.objects.get(
                camera__uuid__iexact=camera_uuid,
                incident_type__key__iexact=incident_type_key,
            )
            camera_incident_type.enabled = False
            camera_incident_type.save()
            _print_done_message()
        else:
            _print_skipped_message()

    def run(self) -> None:
        """Run the sync operation."""
        pairs_to_enable = (
            self._desired_camera_incident_type_pairs
            - self._current_camera_incident_type_pairs
        )

        if pairs_to_enable:
            _print_section_header(
                f"Enable these incident types? ({len(pairs_to_enable)} total)"
            )
            for (camera_uuid, incident_type_key) in pairs_to_enable:
                self._enable_camera_incident_type(
                    camera_uuid, incident_type_key
                )

        pairs_to_disable = (
            self._current_camera_incident_type_pairs
            - self._desired_camera_incident_type_pairs
        )

        if pairs_to_disable:
            _print_section_header(
                f"Disable these incident types? ({len(pairs_to_enable)} total)"
            )
            for (camera_uuid, incident_type_key) in pairs_to_disable:
                self._disable_camera_incident_type(
                    camera_uuid, incident_type_key
                )


if __name__ == "__main__":
    SyncCameras().run()
