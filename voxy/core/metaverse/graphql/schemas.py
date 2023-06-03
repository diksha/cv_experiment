#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import graphene

from core.metaverse.graphql.camera.mutations import (
    CameraConfigMutations,
    CameraMutations,
)
from core.metaverse.graphql.camera.schema import CameraQueries
from core.metaverse.graphql.datacollection.mutations import (
    DataCollectionMutations,
)
from core.metaverse.graphql.datacollection.schema import (
    ActorQueries,
    DataCollectionQueries,
    DataCollectionTestQueries,
)
from core.metaverse.graphql.label.mutations import LabelMutations
from core.metaverse.graphql.label.schema import LabelProjectQueries
from core.metaverse.graphql.model.mutations import ModelMutations
from core.metaverse.graphql.model.schema import (
    DatapoolQueries,
    DatasetQueries,
    LogsetQueries,
    ModelQueries,
    ServiceQueries,
    TaskQueries,
)


class Mutations(
    CameraMutations,
    CameraConfigMutations,
    DataCollectionMutations,
    ModelMutations,
    LabelMutations,
    graphene.ObjectType,
):
    pass


class Query(
    ActorQueries,
    CameraQueries,
    LogsetQueries,
    DataCollectionQueries,
    DataCollectionTestQueries,
    TaskQueries,
    DatasetQueries,
    ServiceQueries,
    ModelQueries,
    DatapoolQueries,
    LabelProjectQueries,
    graphene.ObjectType,
):
    pass


schema = graphene.Schema(query=Query, mutation=Mutations, auto_camelcase=False)
