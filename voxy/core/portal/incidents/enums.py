from enum import Enum

from django.db import models


class IncidentTypeKey(models.TextChoices):
    BUMP_CAP = "BUMP_CAP"
    BAD_POSTURE_WITH_SAFETY_UNIFORM = "BAD_POSTURE_WITH_SAFETY_UNIFORM"
    OVERREACHING_WITH_SAFETY_UNIFORM = "OVERREACHING_WITH_SAFETY_UNIFORM"
    N_PERSON_PED_ZONE = "N_PERSON_PED_ZONE"
    PRODUCTION_LINE_DOWN = "PRODUCTION_LINE_DOWN"
    NO_PED_ZONE = "NO_PED_ZONE"
    NO_STOP_AT_END_OF_AISLE = "NO_STOP_AT_END_OF_AISLE"
    NO_STOP_AT_DOOR_INTERSECTION = "NO_STOP_AT_DOOR_INTERSECTION"
    OVERREACHING = "OVERREACHING"
    Safety_Harness = "Safety_Harness"
    SPILL = "SPILL"
    NO_STOP_AT_INTERSECTION = "NO_STOP_AT_INTERSECTION"
    HARD_HAT = "HARD_HAT"
    SAFETY_VEST = "SAFETY_VEST"
    BAD_POSTURE = "BAD_POSTURE"
    DOOR_VIOLATION = "DOOR_VIOLATION"
    PARKING_DURATION = "PARKING_DURATION"
    PIGGYBACK = "PIGGYBACK"
    OPEN_DOOR_DURATION = "OPEN_DOOR_DURATION"
    MISSING_PPE = "MISSING_PPE"
    HIGH_VIS_HAT_OR_VEST = "HIGH_VIS_HAT_OR_VEST"


class IncidentPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskStatus(Enum):
    OPEN = "open"
    RESOLVED = "resolved"


class ScenarioType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class IncidentCategory(Enum):
    VEHICLE = "VEHICLE"
    ENVIRONMENT = "ENVIRONMENT"
    PEOPLE = "PEOPLE"


class FilterKey(Enum):
    PRIORITY = "PRIORITY"
    STATUS = "STATUS"
    CAMERA = "CAMERA"
    ASSIGNMENT = "ASSIGNMENT"
    INCIDENT_TYPE = "INCIDENT_TYPE"
    EXTRAS = "EXTRAS"

    # NOTE: these are to be only used for offline purposes
    # for internal metrics and tools, NOT for production use
    IS_INVALID = "IS_INVALID"
    IS_VALID = "IS_VALID"
    IS_CORRUPT = "IS_CORRUPT"
    IS_UNSURE = "IS_UNSURE"
    EXCLUDE_UNSURE = "EXCLUDE_UNSURE"
    EXCLUDE_CORRUPT = "EXCLUDE_CORRUPT"
    # ENDNOTE


class StatusFilterOption(Enum):
    UNASSIGNED = "UNASSIGNED_STATUS"
    OPEN_AND_ASSIGNED = "OPEN_AND_ASSIGNED_STATUS"
    RESOLVED = "RESOLVED_STATUS"


class AssignmentFilterOption(Enum):
    ASSIGNED_BY_ME = "ASSIGNED_BY_ME"
    ASSIGNED_TO_ME = "ASSIGNED_TO_ME"


class CooldownSource(models.IntegerChoices):
    """Incident cooldown source."""

    COOLDOWN = 1
    CAMERA_COOLDOWN = 2
    ACTOR_COOLDOWN = 3


class ReviewStatus(models.IntegerChoices):
    """Incident review status."""

    NEEDS_REVIEW = 1
    DO_NOT_REVIEW = 2
    VALID = 3
    INVALID = 4
    # This is used for things like green incidents
    VALID_AND_NEEDS_REVIEW = 5
