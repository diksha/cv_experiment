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


import itertools
import typing
from dataclasses import dataclass

from core.structs.incident import Incident


@dataclass
class IncidentComparison:
    """
    Utility class for incident comparison
    """

    replay_count: typing.Optional[int]
    production_count: typing.Optional[int]
    incident_type: str
    # TODO: see if there are more fields worth adding here


@dataclass
class IncidentComparisonResults:
    """
    Place holder for incident comparison results
    """

    items: typing.List


def incident_differentiator(incident: Incident) -> str:
    """
    Differentiates the incident types

    Args:
        incident (Incident): the incident value

    Returns:
        str: the item to group incidents by (right now just the type)
    """
    return incident.incident_type_id


def aggregate_incident_metrics(
    replay_incidents: typing.List[Incident],
    production_incidents: typing.List[Incident],
) -> IncidentComparisonResults:
    """
    Aggregates and generates a comparison of different incident types

    Args:
        replay_incidents (typing.List[Incident]): the list of incidents
                                             that were generated with replay
        production_incidents (typing.List[Incident]): the production incidents
                                            that were generated in the time interval

    Returns:
        IncidentComparisonResults: the comparison of the replayed incidents to production incidents
    """
    replay_incidents = sorted(replay_incidents, key=incident_differentiator)
    production_incidents = sorted(
        production_incidents, key=incident_differentiator
    )

    grouped_replayed_incidents = itertools.groupby(
        replay_incidents, incident_differentiator
    )
    grouped_production_incidents = itertools.groupby(
        production_incidents, incident_differentiator
    )

    def groups_to_dict_count(groups: tuple) -> dict:
        """
        Generates a dictionary from a grouped set of incidents
        {"incident_type: : count, ...}

        Args:
            groups (tuple): the grouped incidents

        Returns:
            dict: the group of dictionaries
        """
        return {
            group_id: len(list(group_list)) for group_id, group_list in groups
        }

    replayed_incident_counts = groups_to_dict_count(grouped_replayed_incidents)
    production_incident_counts = groups_to_dict_count(
        grouped_production_incidents
    )
    all_incident_types = set(
        list(replayed_incident_counts.keys())
        + list(production_incident_counts.keys())
    )

    return IncidentComparisonResults(
        items=[
            IncidentComparison(
                replay_count=replayed_incident_counts.get(incident_type, 0),
                production_count=production_incident_counts.get(
                    incident_type, 0
                ),
                incident_type=incident_type,
            )
            for incident_type in all_incident_types
        ]
    )
