from google.protobuf.json_format import ParseDict
from google.protobuf.message import Message

from protos.perception.graph_config.v1.graph_params_pb2 import (
    GraphConfig as GraphConfigPb,
)


class GraphConfigValidator:
    def validate_dict_against_graph_config_schema(
        self, input_dict: dict
    ) -> Message:
        """
        Helps determine if a dict fits into the graph config schema

        Args:
            input_dict (dict): The dict of any shape to compare against the graph config schema

        Returns:
            Message: The same message passed as argument, or an error such as ParseError
        """
        return ParseDict(input_dict, GraphConfigPb())


if __name__ == "__main__":
    pass
