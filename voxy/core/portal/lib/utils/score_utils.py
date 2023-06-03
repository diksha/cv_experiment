from core.portal.scores.graphql.types import Score

# Mapping between a ScoreDefinition.name and OrganizationIncidentType.name_override
SCORE_NAME_OVERRIDES: dict[str, dict[str, str]] = {
    "USCOLD": {
        "No Stop at Intersection": "Speeding at Intersection",
        "No Stop at End of Aisle": "Speeding at End of Aisle",
        "No Stop at Door Intersection": "Speeding at Door Intersection",
    }
}


def handle_if_event_score_name_organization_override(
    organization_key: str, event_scores: list[Score]
) -> None:
    """Explicitly checks then handles for the case of an organization-wide override by converting
    event scores to the overrided name.

    Updates applicable event scores with their name_override.
    Otherwise, the given list is not modified

    NOTE: Although this is stored in PDB, the lookup is done with static mapping above
    as this is a very special case. This is not a solution for scale and is only for USCOLD.


    Args:
        organization_key (str): the key of the organization
        event_scores (list[Score]): the event scores with their original ScoreDefinition.name
    """
    score_name_overrides = SCORE_NAME_OVERRIDES.get(organization_key)
    if score_name_overrides:
        for index, score in enumerate(event_scores):
            score_name_override = score_name_overrides.get(score.label)
            if score_name_override:
                event_scores[index] = Score(
                    label=score_name_override, value=score.value
                )
