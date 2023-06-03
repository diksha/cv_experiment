import typing as t

import graphene

from core.portal.accounts.permissions import DOWNTIME_READ
from core.portal.accounts.permissions_manager import has_zone_permission
from core.portal.compliance.graphql.types import ProductionLine
from core.portal.compliance.models.production_line import (
    ProductionLine as ProductionLineModel,
)
from core.portal.lib.graphql.exceptions import PermissionDenied
from core.portal.lib.graphql.utils import pk_from_global_id


class ProductionLineQueries(graphene.ObjectType):
    production_line_details = graphene.Field(
        ProductionLine,
        production_line_id=graphene.ID(),
    )

    def resolve_production_line_details(
        self,
        info: graphene.ResolveInfo,
        production_line_id: t.Optional[str] = None,
    ) -> t.Optional[ProductionLine]:
        """Resolve production line details.

        Args:
            info (graphene.ResolveInfo): graphene context
            production_line_id (t.Optional[str], optional): production line global ID

        Returns:
            t.Optional[ProductionLine]: production line details
        """
        _, production_line_pk = pk_from_global_id(production_line_id)
        production_line = ProductionLineModel.objects.get(
            pk=production_line_pk
        )

        if not has_zone_permission(
            info.context.user,
            production_line.zone,
            DOWNTIME_READ,
        ):
            return PermissionDenied(
                "You do not have permission to view production line details."
            )
        return production_line
