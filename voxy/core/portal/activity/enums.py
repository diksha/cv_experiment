from django.db import models


class SessionScope(models.IntegerChoices):

    UNKNOWN = 0
    GLOBAL = 1
    ORGANIZATION = 2
    SITE = 3
