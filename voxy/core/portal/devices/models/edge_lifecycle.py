from django.contrib import admin
from django.db import models

from core.portal.lib.models.base import Model


class EdgeLifecycle(Model):
    """A lookup table used do store valid edge lifecycles.

    Attributes:
        key (str): The key (name) of the edge lifecycle.
        description (str): The description of the edge lifecycle key.

    Methods:
        __str__(self): Returns the string representation of the edge lifecycle row.
    """

    class Meta:
        app_label = "devices"
        db_table = "edge_lifecycle"

    key = models.CharField(
        help_text="""
        A key denoting the name of the edge lifecycle.
        """,
        max_length=64,
        null=False,
        blank=False,
        unique=True,
    )

    description = models.CharField(
        help_text="A description of the associated edge lifecycle.",
        max_length=256,
        null=False,
        blank=False,
    )

    def __str__(self) -> str:
        """Returns the string representation of the edge lifecycle row.

        Returns:
            str: The string representation of the edge lifecycle row.
        """
        return f"{self.key} - {self.description}"


admin.site.register(EdgeLifecycle)
