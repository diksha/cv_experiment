from django.contrib import admin
from django.db import models

from core.portal.api.models.organization import Organization
from core.portal.lib.models.base import Model
from core.portal.zones.models.zone import Zone


class Role(Model):
    class Meta:
        app_label = "accounts"
        db_table = "role"

    key = models.CharField(
        max_length=250,
        null=False,
        blank=False,
        unique=True,
    )
    visible_to_customers = models.BooleanField(
        null=False,
        blank=False,
        default=False,
    )
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        blank=True,
        null=True,
        related_name="roles",
    )
    zone = models.ForeignKey(
        Zone,
        on_delete=models.CASCADE,
        blank=True,
        null=True,
        related_name="roles",
    )
    name = models.CharField(
        help_text="User friendly name displayed throughout apps.",
        max_length=250,
        null=False,
        blank=False,
    )

    def __str__(self) -> str:
        return f"{self.key} - {self.name}"


class RoleAdmin(admin.ModelAdmin):
    list_display = ("key", "name", "organization", "zone")


admin.site.register(Role, RoleAdmin)
