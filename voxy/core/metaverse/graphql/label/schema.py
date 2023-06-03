#
# Copyright 2023 Voxel Labs, Inc.
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

from core.metaverse.models.label import LabelingTool, LabelProject


class LabelProjectSchema(graphene.ObjectType):
    name = graphene.String()
    description = graphene.String()
    last_checked_timestamp = graphene.DateTime()


class LabelProjectQueries(graphene.ObjectType):
    label_project = graphene.Field(
        LabelProjectSchema,
        labeling_tool=graphene.String(),
        name=graphene.String(),
    )

    def resolve_label_project(self, _, *__, **kwargs) -> LabelProject:
        """Returns given label project from tool and project name

        Args:
            kwargs (Any): keyword arguments for the label project query
            __ (Any): unused positional arguments

        Returns:
            LabelProject: the label project object

        Raises:
            RuntimeError: if the label project cannot be found
        """
        labeling_tool = LabelingTool.nodes.get(name=kwargs["labeling_tool"])
        for project in labeling_tool.project_ref.all():
            if project.name == kwargs["name"]:
                return project

        raise RuntimeError("Could not find project with given name")
