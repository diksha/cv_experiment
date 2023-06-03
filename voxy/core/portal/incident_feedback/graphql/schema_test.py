import graphene
import pytest
from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import Client as DjangoTestClient
from graphene.test import Client as GrapheneTestClient

from core.portal.accounts.permissions import (
    INCIDENT_FEEDBACK_READ,
    REVIEW_QUEUE_READ,
)
from core.portal.api.models.incident_type import IncidentType
from core.portal.api.models.organization import Organization
from core.portal.incident_feedback.graphql.schema import (
    IncidentFeedbackQueries,
)
from core.portal.incidents.enums import ReviewStatus
from core.portal.incidents.models.review_level import ReviewLevel
from core.portal.testing.factories import (
    IncidentFactory,
    IncidentTypeFactory,
    OrganizationFactory,
    UserFactory,
)


class GrapheneContext:
    user = None


@pytest.fixture(name="user")
def _user():
    """User fixture.

    Returns:
        User: user
    """
    return UserFactory(
        username="new_user",
        email="admin1@example.com",
        is_staff=True,
        is_superuser=True,
        permissions=[
            INCIDENT_FEEDBACK_READ.global_permission_key,
            REVIEW_QUEUE_READ.global_permission_key,
        ],
    )


@pytest.fixture(name="incident_type")
def _incident_type():
    """Incident type fixture

    Returns:
        IncidentType: incident type
    """
    return IncidentTypeFactory(key="foo")


@pytest.fixture(name="organization")
def _organization(incident_type: IncidentType):
    """Organization fixture

    Args:
        bad_posture (IncidentType): enabled incident type

    Returns:
        Organization: organization
    """
    return OrganizationFactory(
        key="TPM Shipping & Storage", incident_types=[incident_type]
    )


@pytest.fixture(name="graphene_test_client")
def _graphene_test_client() -> GrapheneTestClient:
    """Graphene test client fixture.

    Returns:
        GrapheneTestClient: graphene test client
    """
    return GrapheneTestClient(
        graphene.Schema(query=IncidentFeedbackQueries, mutation=None)
    )


@pytest.fixture(name="graphene_context")
def _graphene_context(user: User) -> GrapheneContext:
    """Graphene context fixture.

    Args:
        user (User): graphene context user

    Returns:
        GrapheneContext: graphene context
    """
    context = GrapheneContext()
    context.user = user
    return context


@pytest.mark.django_db
def test_incident_feedback_queue_queries_empty(
    client: DjangoTestClient,
    user: User,
    organization: Organization,
    incident_type: IncidentType,
    graphene_test_client: GrapheneTestClient,
    graphene_context: GrapheneContext,
) -> None:
    """Test that incident feedback queue response is empty when queue is empty.

    Args:
        client (DjangoTestClient): django test client
        user (User): test user
        organization (Organization): test organization
        incident_type (IncidentType): test incident type
        graphene_test_client (GrapheneTestClient): graphene test client
        graphene_context (GrapheneContext): graphene context
    """
    cache.clear()
    client.force_login(user)

    # Red incident
    IncidentFactory(
        title="Red incident",
        organization=organization,
        incident_type=incident_type,
        review_level=ReviewLevel.RED,
        review_status=ReviewStatus.DO_NOT_REVIEW,
    )

    # Yellow incident
    IncidentFactory(
        title="Yellow incident",
        organization=organization,
        incident_type=incident_type,
        review_level=ReviewLevel.YELLOW,
        review_status=ReviewStatus.DO_NOT_REVIEW,
    )

    # Green incident with one valid feedback
    IncidentFactory(
        title="Green incident",
        organization=organization,
        incident_type=incident_type,
        review_level=ReviewLevel.GREEN,
        review_status=ReviewStatus.DO_NOT_REVIEW,
    )

    # Gold incident
    IncidentFactory(
        title="Gold incident",
        organization=organization,
        incident_type=incident_type,
        review_level=ReviewLevel.GOLD,
        review_status=ReviewStatus.DO_NOT_REVIEW,
    )
    executed = graphene_test_client.execute(
        """{
            incidentFeedbackQueue(reviewQueueContext: {}) {
                pk
                title
            }
        }
        """,
        context=graphene_context,
    )
    assert executed == {"data": {"incidentFeedbackQueue": []}}


@pytest.mark.django_db
def test_incident_feedback_queue_queries_red_level(
    client: DjangoTestClient,
    user: User,
    organization: Organization,
    incident_type: IncidentType,
    graphene_test_client: GrapheneTestClient,
    graphene_context: GrapheneContext,
) -> None:
    """Test that incident feedback queue returns red incidents which require more reviews.

    Args:
        client (DjangoTestClient): django test client
        user (User): test user
        organization (Organization): test organization
        incident_type (IncidentType): test incident type
        graphene_test_client (GrapheneTestClient): graphene test client
        graphene_context (GrapheneContext): graphene context
    """
    cache.clear()
    client.force_login(user)
    # Added to review queue
    IncidentFactory(
        title="Red incident only one review",
        organization=organization,
        incident_type=incident_type,
        review_level=ReviewLevel.RED,
    )

    executed = graphene_test_client.execute(
        """{
            incidentFeedbackQueue(reviewQueueContext: {}) {
                pk
                title
            }
        }
        """,
        context=graphene_context,
    )
    assert executed == {
        "data": {
            "incidentFeedbackQueue": [
                {"pk": 1, "title": "Red incident only one review"}
            ]
        }
    }


@pytest.mark.django_db
def test_incident_feedback_queue_queries_yellow_level(
    client: DjangoTestClient,
    user: User,
    organization: Organization,
    incident_type: IncidentType,
    graphene_test_client: GrapheneTestClient,
    graphene_context: GrapheneContext,
) -> None:
    """Test that incident feedback queue returns yellow incidents which require more reviews.

    Args:
        client (DjangoTestClient): django test client
        user (User): test user
        organization (Organization): test organization
        incident_type (IncidentType): test incident type
        graphene_test_client (GrapheneTestClient): graphene test client
        graphene_context (GrapheneContext): graphene context
    """
    cache.clear()
    client.force_login(user)
    # Added to review queue
    IncidentFactory(
        title="Yellow incident no review",
        organization=organization,
        incident_type=incident_type,
        review_level=ReviewLevel.YELLOW,
    )
    executed = graphene_test_client.execute(
        """{
            incidentFeedbackQueue(reviewQueueContext: {}) {
                pk
                title
            }
        }
        """,
        context=graphene_context,
    )
    assert executed == {
        "data": {
            "incidentFeedbackQueue": [
                {"pk": 1, "title": "Yellow incident no review"}
            ]
        }
    }


@pytest.mark.django_db
def test_incident_feedback_queue_queries_green_level(
    client: DjangoTestClient,
    user: User,
    organization: Organization,
    incident_type: IncidentType,
    graphene_test_client: GrapheneTestClient,
    graphene_context: GrapheneContext,
) -> None:
    """Test that incident feedback queue returns green incidents which require more reviews.

    Args:
        client (DjangoTestClient): django test client
        user (User): test user
        organization (Organization): test organization
        incident_type (IncidentType): test incident type
        graphene_test_client (GrapheneTestClient): graphene test client
        graphene_context (GrapheneContext): graphene context
    """
    cache.clear()
    client.force_login(user)
    # Added to review queue
    incident = IncidentFactory(
        title="Green incident no review",
        organization=organization,
        incident_type=incident_type,
        review_level=ReviewLevel.GREEN,
    )

    # force review status
    incident.review_status = ReviewStatus.VALID_AND_NEEDS_REVIEW
    incident.save()

    executed = graphene_test_client.execute(
        """{
            incidentFeedbackQueue(reviewQueueContext: {}) {
                pk
                title
            }
        }
        """,
        context=graphene_context,
    )
    assert executed == {
        "data": {
            "incidentFeedbackQueue": [
                {"pk": 1, "title": "Green incident no review"}
            ]
        }
    }
