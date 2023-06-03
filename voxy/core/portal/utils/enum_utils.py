from ast import Dict
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from core.portal.devices.models.edge_lifecycle import (
    EdgeLifecycle as EdgeLifecycleModel,
)
from core.portal.lib.models.base import Model


@dataclass(frozen=True)
class EnumConfig:
    """Standard key and description for a lookup table referencing a static enum

    Attributes:
        key (str): The key of the static enum
        description (str): The description of the static enum
    """

    key: str
    description: str


def fetch_enum_model_from_name(name: str) -> Model:
    """Fetches the enum model from the name of the enum.

    Args:
        name (str): The name of the enum.

    Returns:
        Model: The enum model.

    Raises:
        ValueError: If the enum model cannot be found.
    """

    if name == "EdgeLifecycle":
        return EdgeLifecycleModel

    raise ValueError(f"Unknown enum name: {name}")


def sync_static_enum(
    lookup_table_model: Model, enum_config_map: Dict(Enum, EnumConfig)
):
    """Ensures that the static roles exist in a database.
    Args:
        lookup_table_model (Model): The lookup table (database representation of an enum)
                                    model to sync.
        enum_config_map (Dict[Enum, EnumConfig]): The static enum configuration to sync.

    Raises:
        Exception: If the static roles cannot be created
    """

    logger.info(
        f"syncing lookup table {str(lookup_table_model).rsplit('.', maxsplit=1)[-1][0:-2]}"
    )

    for value in enum_config_map.values():
        logger.info(f"Syncing enum key: {value.key}")
        state = lookup_table_model.objects.filter(key=value.key).first()
        if state:
            logger.debug(f"Enum key exists: {value.key}")

        else:
            logger.info(f"Creating static enum key: {value.key}")
            try:
                state = lookup_table_model.objects.create(
                    key=value.key,
                    description=value.description,
                )
            except Exception as exc:
                logger.error(f"Failed to create enum key: {value.key}: {exc}")
                raise exc

        logger.debug(f"Syncing enum key {value.key} complete")

        logger.debug(
            f"syncing lookup table \
                {str(lookup_table_model).rsplit('.', maxsplit=1)[-1][0:-2]} is complete"
        )
