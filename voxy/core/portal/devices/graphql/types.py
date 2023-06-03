import typing as t

import graphene
from graphene_django import DjangoObjectType

from core.portal.devices.models.camera import Camera as CameraModel
from core.portal.devices.models.camera import (
    CameraConfigNew as CameraConfigNewModel,
)
from core.portal.devices.models.edge import Edge as EdgeModel

if t.TYPE_CHECKING:
    from core.portal.incidents.graphql.types import CameraIncidentType


class CameraConfigNewModelType(DjangoObjectType):
    """The CameraConfigNewModel type used for graphql queries"""

    class Meta:
        model = CameraConfigNewModel
        fields = "__all__"


# TODO: replace with Django model type
class CameraType(DjangoObjectType):
    """The Camera type used for graphql queries"""

    class Meta:
        model = CameraModel
        interfaces = [graphene.relay.Node]

    thumbnail_url = graphene.String()
    incident_types = graphene.List(
        graphene.NonNull(
            graphene.lazy_import(
                "core.portal.incidents.graphql.types.CameraIncidentType"
            )
        ),
        required=True,
    )

    @staticmethod
    def resolve_incident_types(
        parent: CameraModel,
        _: graphene.ResolveInfo,
    ) -> t.List["CameraIncidentType"]:
        """Resolve camera incident types

        Args:
            parent (CameraModel): camera model instance

        Returns:
            t.List[CameraIncidentType]: list of camera incident types
        """
        return parent.camera_incident_types.filter(enabled=True)


class EdgeType(DjangoObjectType):
    """The Edge type used for graphql queries"""

    class Meta:
        model = EdgeModel
        fields = "__all__"
        interfaces = [graphene.relay.Node]
