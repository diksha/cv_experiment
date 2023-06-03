import typing as t
from datetime import datetime, timedelta

import graphene

from core.portal.accounts.permissions import PAGE_DASHBOARD
from core.portal.accounts.permissions_manager import has_zone_permission
from core.portal.lib.graphql.exceptions import PermissionDenied
from core.portal.scores.services import get_site_event_scores
from core.portal.zones.models.zone import Zone


class Score(graphene.ObjectType):
    label = graphene.String(required=True)
    value = graphene.Int(required=True)


class SiteEventScore(graphene.ObjectType):
    # TODO: add config field
    label = graphene.String(required=True)
    score = graphene.Int(required=True)


class SiteScoreStats(graphene.ObjectType):
    def __init__(self, site: Zone) -> None:
        super().__init__()
        self.site = site

    # TODO: add config field
    site_score = graphene.Int()

    @staticmethod
    def resolve_site_score(
        parent: "SiteScoreStats", info: graphene.ResolveInfo
    ) -> t.Optional[int]:
        """Retrieves the site-wide score

        Args:
            parent (SiteScoreStats): Site
            info (graphene.ResolveInfo): graphene context

        Returns:
            t.Optional[int]: Returns the site-wide score, if exists
        """
        # TODO: need to figure out how to take average of the resolve_event_scores
        return 0

    site_event_scores = graphene.List(graphene.NonNull(SiteEventScore))

    @staticmethod
    def resolve_site_event_scores(
        parent: "SiteScoreStats", info: graphene.ResolveInfo
    ) -> list[SiteEventScore]:
        """Retrieves all the event scores for the site

        Args:
            parent (SiteScoreStats): Site
            info (graphene.ResolveInfo): graphene context

        Returns:
            list[SiteEventScore]: list of site-event scores
        """
        # TODO: Add permission check for 'SiteScoreCard'
        if not parent.site or not has_zone_permission(
            info.context.user,
            parent.site,
            PAGE_DASHBOARD,
        ):
            return PermissionDenied(
                "You do not have permission to view dashboard (site-score)."
            )

        date_time = datetime.now() - timedelta(days=1)
        data = get_site_event_scores(site=parent.site, date_time=date_time)

        event_scores = [
            SiteEventScore(label=key, score=value)
            for key, value in data.items()
            if key and value is not None
        ]

        return event_scores
