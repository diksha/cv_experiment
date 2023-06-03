from enum import Enum


class QueryModelType(Enum):
    """The supported filter Model types.

    Args:
      Enum: enum type
    """

    DOOR_EVENT_AGGREGATE = "DoorEventAggregate"
    DOOR_OPEN_AGGREGATE = "DoorOpenAggregate"
    ERGONOMICS_AGGREGATE = "ErgonomicsAggregate"
    EVENT = "Event"
    INCIDENT = "Incident"
    PRODUCTION_LINE = "ProductionLine"
    STATE = "State"
