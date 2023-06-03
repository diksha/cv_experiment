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

from typing import Any

import graphene
from neomodel import db

from core.metaverse.graphql.label.schema import LabelProjectSchema
from core.metaverse.models.label import LabelingTool, LabelProject


class LabelProjectCreate(graphene.Mutation):
    class Arguments:
        labeling_tool = graphene.String(required=True)
        name = graphene.String(required=True)
        description = graphene.String(default_value="")

    success = graphene.Boolean()
    label_project = graphene.Field(LabelProjectSchema)
    error = graphene.String()

    def mutate(
        self,
        info: graphene.ResolveInfo,
        **kwargs: Any,
    ) -> "LabelProjectCreate":
        """Creates a new label project in metaverse

        Args:
            info (graphene.ResolveInfo): the resolver info for graphql
            kwargs (Any): keyword arguments for the label project creation mutation

        Raises:
            RuntimeError: if the label project already exists

        Returns:
            LabelProjectCreate: The label project creation object
        """
        db.begin()
        try:
            labeling_tool, name, description = (
                kwargs["labeling_tool"],
                kwargs["name"],
                kwargs.get("description"),
            )
            label_tool = LabelingTool.nodes.get(name=labeling_tool)
            for project in label_tool.project_ref.all():
                if project.name == name:
                    raise RuntimeError(
                        f"Project with name {name} already exists"
                    )
            label_project = LabelProject(
                name=name, description=description
            ).save()
            label_tool.project_ref.connect(label_project)
        # trunk-ignore(pylint/W0718)
        except Exception as error:
            db.rollback()
            return LabelProjectCreate(
                label_project=None, success=False, error=error
            )
        db.commit()
        success = True
        return LabelProjectCreate(label_project=label_project, success=success)


class LabelProjectUpdate(graphene.Mutation):
    class Arguments:
        labeling_tool = graphene.String(required=True)
        name = graphene.String(required=True)
        updated_time = graphene.DateTime(required=True)

    success = graphene.Boolean()
    label_project = graphene.Field(LabelProjectSchema)
    error = graphene.String()

    def mutate(
        self,
        info: graphene.ResolveInfo,
        **kwargs: Any,
    ) -> "LabelProjectUpdate":
        """Updates a label project in metaverse with checkpoint

        Args:
            info (graphene.ResolveInfo): the resolver info for graphql
            kwargs (Any): keyword arguments for the label project update mutation

        Returns:
            LabelProjectUpdate: The label project updation object
        """
        db.begin()
        try:
            project = None
            labeling_tool, name, updated_time = (
                kwargs["labeling_tool"],
                kwargs["name"],
                kwargs["updated_time"],
            )
            label_tool = LabelingTool.nodes.get(name=labeling_tool)
            for project in label_tool.project_ref.all():
                if project.name == name:
                    project.last_checked_timestamp = updated_time
                    project.save()
                    break
        # trunk-ignore(pylint/W0718)
        except Exception as error:
            db.rollback()
            return LabelProjectUpdate(
                label_project=None, success=False, error=error
            )
        db.commit()
        success = True
        return LabelProjectUpdate(label_project=project, success=success)


class LabelMutations(graphene.ObjectType):
    """Label mutations for metaverse

    Args:
        graphene (graphene.ObjectType): The base object type for graphene
    """

    label_project_create = LabelProjectCreate.Field()
    label_project_update = LabelProjectUpdate.Field()
