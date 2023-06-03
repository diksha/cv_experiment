from django.contrib import admin
from django.contrib.auth.models import User
from django.db import models

from core.portal.accounts.models.role import Role
from core.portal.lib.models.base import Model


class UserRole(Model):
    class Meta:
        app_label = "accounts"
        db_table = "user_role"

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        blank=False,
        null=False,
        related_name="user_roles",
    )
    role = models.ForeignKey(
        Role,
        on_delete=models.CASCADE,
        blank=False,
        null=False,
        related_name="user_roles",
    )
    assigned_at = models.DateTimeField(
        blank=False,
        null=False,
        auto_now_add=True,
    )
    assigned_by = models.ForeignKey(
        User,
        # Don't allow deleting users if they're part of an audit trail
        on_delete=models.RESTRICT,
        blank=True,
        null=True,
        related_name="user_roles_assigned_by",
    )
    removed_at = models.DateTimeField(
        blank=True,
        null=True,
    )
    removed_by = models.ForeignKey(
        User,
        # Don't allow deleting users if they're part of an audit trail
        on_delete=models.RESTRICT,
        blank=True,
        null=True,
        related_name="user_roles_removed_by",
    )


class UserRoleAdmin(admin.ModelAdmin):
    list_display = ("user", "role")


admin.site.register(UserRole, UserRoleAdmin)
