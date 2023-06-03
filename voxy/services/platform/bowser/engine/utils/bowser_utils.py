import os
import sys

from loguru import logger
from rules_python.python.runfiles import runfiles


class BowserUtils:
    @staticmethod
    def try_set_bowser_home():
        """Setting up mandatory env variable to make work Bowser."""

        if os.getenv("FLINK_HOME") is not None:
            return
        try:
            runfiles_dir = runfiles.Create().EnvVars()["RUNFILES_DIR"]
            bowser_libraries = os.path.join(
                runfiles_dir,
                "pip_deps_apache_flink_libraries/site-packages/pyflink",
            )

            if not os.path.exists(bowser_libraries):
                logger.warning("could not find bowser libraries")
                return

            os.environ["FLINK_LIB_DIR"] = os.path.join(bowser_libraries, "lib")
            os.environ["FLINK_OPT_DIR"] = os.path.join(bowser_libraries, "opt")
            os.environ["FLINK_PLUGINS_DIR"] = os.path.join(
                bowser_libraries, "plugins"
            )

        # trunk-ignore(pylint/W0718): this is best effort and should fail with only a warning
        except Exception as ex:
            logger.warning("failed to set bowser env vars: %s", ex)

    @staticmethod
    def set_loger(log_level: str):

        logger.remove()
        logger.add(
            sys.stderr,
            colorize=True,
            format="<green>{time}</green> | <blue>{level}</blue> |  {message}",
            level=log_level,
        )
