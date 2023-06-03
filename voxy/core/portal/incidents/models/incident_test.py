import pytest

from core.portal.api.models.incident import Incident, UserIncident
from core.portal.incidents.enums import AssignmentFilterOption, FilterKey
from core.portal.incidents.models.review_level import ReviewLevel
from core.portal.incidents.types import Filter
from core.portal.testing.factories import (
    CameraConfigNewFactory,
    CameraFactory,
    IncidentFactory,
    IncidentTypeFactory,
    OrganizationFactory,
    UserFactory,
)


@pytest.mark.django_db
def test_default_manager_excludes_experimental_incidents() -> None:
    """Test that default model manager excludes experimental incidents."""
    IncidentFactory(experimental=False)
    IncidentFactory(experimental=True)
    IncidentFactory(experimental=True)
    IncidentFactory(experimental=True)
    assert Incident.objects.count() == 1


@pytest.mark.django_db
def test_experimental_manager_includes_experimental_incidents() -> None:
    """Test that experimental model manager includes experimental incidents."""
    IncidentFactory(experimental=False)
    IncidentFactory(experimental=True)
    IncidentFactory(experimental=True)
    IncidentFactory(experimental=True)
    assert Incident.objects_experimental.count() == 3


@pytest.mark.django_db
def test_data_backed_properties() -> None:
    """Test that properties backed by data field return values from data field."""
    IncidentFactory(
        data={
            "video_s3_path": "video_s3_path",
            "annotations_s3_path": "annotations_s3_path",
            "actor_ids": "1,2,3,4,foo,bar",
            "incident_version": "incident_version",
        }
    )
    incident = Incident.objects_raw.first()

    assert incident.video_s3_path == "video_s3_path"
    assert incident.annotations_s3_path == "annotations_s3_path"
    assert incident.actor_ids == ["1", "2", "3", "4", "foo", "bar"]
    assert incident.incident_version == "incident_version"


@pytest.mark.django_db
def test_review_level_default() -> None:
    """Test review level defaults to red."""
    IncidentFactory()
    incident = Incident.objects_raw.first()
    assert incident.review_level == ReviewLevel.RED


@pytest.mark.django_db
def test_data_camera_config() -> None:
    """Test incident camera config happy path."""
    organization = OrganizationFactory(key="key")
    camera = CameraFactory(uuid="uuid", name="name", organization=organization)
    CameraConfigNewFactory(
        camera=camera,
        version=1,
    )
    IncidentFactory(
        camera=camera,
        data={
            "camera_config_version": "1",
        },
    )
    incident = Incident.objects_raw.first()

    assert incident.camera_config.version == 1
    assert incident.camera_config.camera.uuid == "uuid"


@pytest.mark.django_db
def test_review_level_green() -> None:
    """Test incident review level happy path."""
    IncidentFactory(review_level=ReviewLevel.GREEN)
    incident = Incident.objects_raw.first()
    assert incident.review_level == ReviewLevel.GREEN


# TODO(diksha) resolve jsondecodeerror for F("json_field")
# @pytest.mark.django_db
# def test_filter_for_organization() -> None:
#     organization = OrganizationFactory(key="TPM Shipping & Storage")
#     incident_type = IncidentTypeFactory(key="foo")
#     OrganizationIncidentTypeFactory(
#         organization=organization,
#         incident_type=incident_type,
#         enabled=True,
#     )
#     # Shown to customers
#     IncidentFactory(
#         organization=organization,
#         incident_type=incident_type,
#         review_level=ReviewLevel.RED,
#         valid_feedback_count=1,
#         invalid_feedback_count=0,
#     )
#     IncidentFactory(
#         organization=organization,
#         incident_type=incident_type,
#         review_level=ReviewLevel.GREEN,
#         valid_feedback_count=0,
#         invalid_feedback_count=0,
#     )

#     # Not shown to customers
#     IncidentFactory(
#         organization=organization,
#         incident_type=incident_type,
#         review_level=ReviewLevel.RED,
#         valid_feedback_count=0,
#         invalid_feedback_count=0,
#     )
#     IncidentFactory(
#         organization=organization,
#         incident_type=incident_type,
#         review_level=ReviewLevel.RED,
#         valid_feedback_count=1,
#         invalid_feedback_count=1,
#     )

#     incident = Incident.objects.for_organization(organization)
#     assert incident.count() == 2


@pytest.mark.django_db
def test_priority_filter() -> None:
    """Test that priority filter returns expected values."""
    for _ in range(3):
        IncidentFactory(priority="HIGH")
    for _ in range(2):
        IncidentFactory(priority="MEDIUM")
    for _ in range(1):
        IncidentFactory(priority="LOW")

    high_filters = [Filter(FilterKey.PRIORITY.value, "HIGH")]
    assert Incident.objects.apply_filters(high_filters, None).count() == 3

    medium_filters = [Filter(FilterKey.PRIORITY.value, "MEDIUM")]
    assert Incident.objects.apply_filters(medium_filters, None).count() == 2

    low_filters = [Filter(FilterKey.PRIORITY.value, "LOW")]
    assert Incident.objects.apply_filters(low_filters, None).count() == 1

    unknown_filters = [Filter(FilterKey.PRIORITY.value, "UNKNOWN")]
    assert Incident.objects.apply_filters(unknown_filters, None).count() == 0


@pytest.mark.django_db
def test_incident_type_filter() -> None:
    """Test that incident type filter returns expected values."""
    IncidentFactory(incident_type=IncidentTypeFactory(key="foo"))
    IncidentFactory(incident_type=IncidentTypeFactory(key="foo"))
    IncidentFactory(incident_type=IncidentTypeFactory(key="bar"))

    foo_filters = [Filter(FilterKey.INCIDENT_TYPE.value, "foo")]
    assert Incident.objects.apply_filters(foo_filters, None).count() == 2

    bar_filters = [Filter(FilterKey.INCIDENT_TYPE.value, "bar")]
    assert Incident.objects.apply_filters(bar_filters, None).count() == 1

    buzz_filters = [Filter(FilterKey.INCIDENT_TYPE.value, "buzz")]
    assert Incident.objects.apply_filters(buzz_filters, None).count() == 0


@pytest.mark.django_db
def test_assignment_filter() -> None:
    """Test that assignment filter returns expected values."""
    alice = UserFactory(username="alice")
    bob = UserFactory(username="bob")

    for _ in range(3):
        incident = IncidentFactory()
        UserIncident.objects.create(
            incident=incident, assigned_by=alice, assignee=bob
        )

    assigned_by_alice_filters = [
        Filter(
            FilterKey.ASSIGNMENT.value,
            AssignmentFilterOption.ASSIGNED_BY_ME.value,
        )
    ]
    assert (
        Incident.objects.apply_filters(
            assigned_by_alice_filters, alice
        ).count()
        == 3
    )

    assigned_to_alice_filters = [
        Filter(
            FilterKey.ASSIGNMENT.value,
            AssignmentFilterOption.ASSIGNED_TO_ME.value,
        )
    ]
    assert (
        Incident.objects.apply_filters(
            assigned_to_alice_filters, alice
        ).count()
        == 0
    )

    assigned_by_bob_filters = [
        Filter(
            FilterKey.ASSIGNMENT.value,
            AssignmentFilterOption.ASSIGNED_BY_ME.value,
        )
    ]
    assert (
        Incident.objects.apply_filters(assigned_by_bob_filters, bob).count()
        == 0
    )

    assigned_to_bob_filters = [
        Filter(
            FilterKey.ASSIGNMENT.value,
            AssignmentFilterOption.ASSIGNED_TO_ME.value,
        )
    ]
    assert (
        Incident.objects.apply_filters(assigned_to_bob_filters, bob).count()
        == 3
    )
