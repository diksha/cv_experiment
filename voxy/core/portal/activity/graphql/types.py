import graphene

from core.portal.accounts.graphql.types import UserType


class SessionUserCount(graphene.ObjectType):
    user = graphene.Field(UserType)
    value = graphene.Int(required=True)


class SessionSiteCount(graphene.ObjectType):
    site = graphene.Field(
        graphene.lazy_import("core.portal.zones.graphql.types.ZoneType")
    )
    value = graphene.Int(required=True)


class SessionCount(graphene.ObjectType):
    users = graphene.List(SessionUserCount)
    sites = graphene.List(SessionSiteCount)
