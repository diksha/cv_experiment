import os
import unittest

import yaml

from services.platform.polygon.lib.proto_schema.validate_graph_config_schema import (
    GraphConfigValidator,
)


class YamlProtobufTest(unittest.TestCase):
    def get_yaml_from_file(self, file_location: str) -> dict:
        """
        Helper func to turn a yaml file into a dict

        Args:
            file_location (str): path to yaml file

        Returns:
            dict: yaml in dict form
        """
        with open(file_location, "r", encoding="utf-8") as yaml_in:
            return yaml.safe_load(yaml_in)

    def test_all_cha_yaml_matches_protobuf_schema(self) -> None:
        rootdir = os.path.join(os.getcwd(), "configs/cameras")
        for subdir, _dirs, files in os.walk(rootdir):
            for file in files:
                if file == "cha.yaml":
                    full_path = f"{subdir}/{file}"
                    yaml_object = self.get_yaml_from_file(full_path)
                    GraphConfigValidator().validate_dict_against_graph_config_schema(
                        yaml_object
                    )

    def test_default_and_env_configs_matches_protobuf_schema(self) -> None:
        files_to_check = [
            "configs/graphs/default.yaml",
            "configs/graphs/production/environment/production.yaml",
            "configs/graphs/production/environment/development.yaml",
        ]
        for file in files_to_check:
            file_location = os.path.join(os.getcwd(), file)
            yaml_object = self.get_yaml_from_file(file_location)
            GraphConfigValidator().validate_dict_against_graph_config_schema(
                yaml_object
            )


if __name__ == "__main__":
    unittest.main()
