import argparse

ENVIRONMENT_PROD = "production"
ENVIRONMENT_DEV = "development"
RUNTIME_ENVIRONMENTS = [ENVIRONMENT_DEV, ENVIRONMENT_PROD]


class BowserConfigUtils:
    @staticmethod
    def parse_args(default_runtime_environment: str) -> argparse.Namespace:
        """Parse arguments from CLI

        :param str default_runtime_environment: default runtime

        :returns: parsed command line configuration
        :rtype: argparse.Namespace
        """
        parser = argparse.ArgumentParser(
            description="Production Graph Runner", allow_abbrev=False
        )
        parser.add_argument("--bowser_config_path", type=str, required=False)
        parser.add_argument(
            "--environment",
            type=str,
            choices=RUNTIME_ENVIRONMENTS,
            default=default_runtime_environment,
        )
        parser.add_argument("--logging_level", type=str, default="info")
        _args, _ = parser.parse_known_args()
        return _args
