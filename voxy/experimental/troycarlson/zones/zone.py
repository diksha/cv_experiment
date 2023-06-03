import json

from django.db import models

from core.portal.api.models.organization import Organization
from core.portal.lib.models.base import Model
from core.portal.zones.node import CameraLocationTree


class Zone(Model):
    class Meta:
        app_label = "zones"
        db_table = "zones"

    class ZoneTypeName(models.TextChoices):
        SITE = "site"
        ROOM = "room"
        AREA = "area"

    parent_zone = models.ForeignKey(
        "self",
        on_delete=models.CASCADE,
        blank=True,
        null=True,
        max_length=250,
    )

    name = models.CharField(
        max_length=250,
        null=True,
        blank=False,
    )

    zone_type = models.CharField(
        max_length=10,
        choices=ZoneTypeName.choices,
        default=ZoneTypeName.SITE,
    )

    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        null=True,
        related_name="zones",
    )

    # test functions to return a whole tree from the node

    @classmethod
    def self_test(cls, node_id, org, display_all):
        relationship_pairs = cls.objects.filter(organization=org).values_list()

        tree = CameraLocationTree()
        for data in relationship_pairs:
            tree.insert(data[1:])

        resp = tree.get_tree(org, node_id)
        if display_all == "1":
            resp = json.dumps(resp)
        else:
            resp = [resp.keys()]

        return resp

    @classmethod
    def child(cls, node_id, org):
        relationship_pairs = cls.objects.filter(organization=org).values_list()

        tree = CameraLocationTree()
        for data in relationship_pairs:
            tree.insert(data[1:])

        resp = tree.get_tree(org, node_id).keys()

        return str(resp)

    def __str__(self) -> str:
        return f"Zone is #{self.name}"


######################################
