from django.db import models


class PerceivedActorStateDurationCategory(models.IntegerChoices):
    """Perceived actor state duration categories."""

    PERSON_TIME = 1
    PIT_TIME = 2
    PIT_STATIONARY_TIME = 3
    PIT_NON_STATIONARY_TIME = 4


# TODO: Let's move this into the PerceivedEventRateDefinition class
class PerceivedEventRateCalculationMethod(models.IntegerChoices):
    # TODO: need to figure out how to differentiate methods with different group_bys
    # i.e. by camera or by zone. Not super urgent but will get confusing
    HOURLY_DISCRETE = 1
    HOURLY_CONTINUOUS = 2
    THIRTY_DAY_DISCRETE = 3
    THIRTY_DAY_CONTINUOUS = 4
