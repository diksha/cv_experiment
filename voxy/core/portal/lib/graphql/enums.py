import graphene

from core.portal.lib.enums import TimeBucketWidth as TimeBucketWidthEnum

TimeBucketWidth = graphene.Enum.from_enum(TimeBucketWidthEnum)
