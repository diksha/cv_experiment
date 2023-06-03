import sys
import typing

from loguru import logger

LOG_LEVEL_TRACE = "TRACE"
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_SUCCESS = "SUCCESS"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"
LOG_LEVEL_CRITICAL = "CRITICAL"

ALL_LOGGING_LEVELS = [
    LOG_LEVEL_TRACE,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_SUCCESS,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_CRITICAL,
]


def configure_logger(
    extra: typing.Optional[dict] = None,
    level: str = LOG_LEVEL_INFO,
    serialize: bool = False,
) -> None:
    """Set logger configuration for loguru

    Args:
        extra (typing.Optional[dict], optional): Any additional configs. Defaults to None.
        level (str, optional): verbosity of logger. Defaults to LOG_LEVEL_INFO.
        serialize (bool, optional): Set to true to produce JSON logs. Defaults to False.
    """
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "backtrace": True,
                "diagnose": True,
                "enqueue": True,
                "colorize": False,
                "serialize": serialize,
                "level": level,
            },
        ],
        "extra": extra or {},
    }

    logger.configure(**config)
