import os
from typing import Optional

import graphene
from django.conf import settings
from graphene_django import DjangoObjectType
from loguru import logger

from core.portal.accounts.graphql.types import UserType
from core.portal.accounts.helpers import get_fullname
from core.portal.accounts.permissions import (
    INCIDENT_PUBLIC_LINK_CREATE,
    INCIDENTS_HIGHLIGHT,
    INCIDENTS_RESOLVE,
)
from core.portal.accounts.permissions_manager import (
    has_global_permission,
    has_zone_permission,
)
from core.portal.api.models.comment import Comment
from core.portal.api.models.incident import Incident, UserIncident
from core.portal.api.models.share_link import ShareLink
from core.portal.incidents.graphql.enums import ScenarioTypeEnum
from core.portal.incidents.graphql.types import IncidentType
from core.portal.incidents.graphql.utils import (
    StorageProvider,
    attachment_disposition,
    log_action,
    video_rendering,
)
from core.portal.lib.graphql.exceptions import PermissionDenied
from core.portal.lib.graphql.mutations import BaseMutation
from core.portal.lib.graphql.utils import pk_from_global_id
from core.portal.notifications.clients.sendgrid import (
    ASSIGN_EMAIL_TEMPLATE_ID,
    INCIDENT_RESOLVED_TEMPLATE_ID,
    SendGridClient,
)
from core.utils import aws_utils


class AssignSchema(DjangoObjectType):
    class Meta:
        model = UserIncident
        fields = "__all__"


class UserIncidentSchema(graphene.InputObjectType):
    incident_id: graphene.ID = graphene.ID(required=True)
    assignee_ids = graphene.List(graphene.ID)
    note = graphene.String()


class CreateUserIncident(BaseMutation):
    class Arguments:
        assign_data = UserIncidentSchema(required=True)

    users = graphene.List(UserType)

    @staticmethod
    def mutate(
        root: None,
        info: graphene.ResolveInfo,
        assign_data: UserIncidentSchema = None,
    ) -> "CreateUserIncident":
        del root
        if not assign_data:
            raise RuntimeError("assign_data is required")
        _, incident_pk = pk_from_global_id(assign_data.incident_id)
        assignee_ids = assign_data.assignee_ids
        note = assign_data.note

        list_saved = []
        for assignee_id in assignee_ids:
            _, assignee_pk = pk_from_global_id(assignee_id)
            instance, _ = UserIncident.objects.get_or_create(
                incident_id=incident_pk,
                assignee_id=assignee_pk,
                defaults=dict(
                    organization_id=info.context.user.profile.current_organization.id,
                    assigned_by=info.context.user,
                    note=note,
                ),
            )
            list_saved.append(instance)

        users = [item.assignee for item in list_saved]

        # Send email
        if settings.SEND_TRANSACTIONAL_EMAILS:
            for instance in list_saved:
                user_name = f"{info.context.user.first_name} {info.context.user.last_name}"
                incident_url = (
                    f"{info.context.scheme}://{info.context.get_host()}/incidents/{instance.incident.uuid}",
                )
                SendGridClient().send_email_with_template(
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    to_emails=[instance.assignee.email],
                    subject=f"{user_name} assigned you to an incident: {instance.incident.title}",
                    template_id=ASSIGN_EMAIL_TEMPLATE_ID,
                    user_who_assign=user_name,
                    incident_title=instance.incident.title,
                    incident_url=incident_url,
                    note=note,
                )

        return CreateUserIncident(users=users)


class DeleteUserIncident(BaseMutation):
    class Arguments:
        incident_id = graphene.ID()
        assignee_id = graphene.ID()

    assign = graphene.Field(AssignSchema)

    @staticmethod
    def mutate(
        root: None,
        info: graphene.ResolveInfo,
        incident_id: str,
        assignee_id: str,
    ) -> "DeleteUserIncident":
        del root
        _, incident_pk = pk_from_global_id(incident_id)
        _, assignee_pk = pk_from_global_id(assignee_id)
        UserIncident.objects.filter(
            organization_id=info.context.user.profile.current_organization.id,
            incident_id=incident_pk,
            assignee_id=assignee_pk,
        ).delete()
        return DeleteUserIncident()


class IncidentCreateScenario(BaseMutation):
    class Arguments:
        incident_id = graphene.ID()
        scenario_type = graphene.Argument(ScenarioTypeEnum)

    incident = graphene.Field(IncidentType)

    @staticmethod
    def mutate(
        root: None,
        info: graphene.ResolveInfo,
        incident_id: str,
        scenario_type: ScenarioTypeEnum,
    ) -> "IncidentCreateScenario":
        del root
        if not info.context.user.is_superuser:
            raise PermissionDenied("Only superusers may create scenarios")
        _, pk = pk_from_global_id(incident_id)
        incident = Incident.objects_raw.get(pk=pk)
        incident.create_scenario(scenario_type.value)
        return IncidentCreateScenario(incident=incident)


class IncidentResolve(BaseMutation):
    class Arguments:
        incident_id = graphene.ID(required=True)

    incident = graphene.Field(IncidentType)

    @classmethod
    def send_email(
        cls, info: graphene.ResolveInfo, incident: Incident
    ) -> None:
        if settings.SEND_TRANSACTIONAL_EMAILS:
            user_name = get_fullname(info.context.user)
            incident_url = (
                f"{info.context.scheme}://{info.context.get_host()}/incidents/{incident.uuid}",
            )

            to_emails = []
            for assigned_by in incident.assigned_by.all():
                if not assigned_by.email:
                    print(f"User {assigned_by.id} has no email.")
                    continue

                to_emails.append(assigned_by.email)

            if len(to_emails) == 0:
                return

            SendGridClient().send_email_with_template(
                from_email=settings.DEFAULT_FROM_EMAIL,
                to_emails=to_emails,
                subject=f"{user_name} resolved an incident: {incident.title}",
                template_id=INCIDENT_RESOLVED_TEMPLATE_ID,
                assignee_name=user_name,
                incident_title=incident.title,
                incident_url=incident_url,
            )

    @staticmethod
    def mutate(
        root: None, info: graphene.ResolveInfo, incident_id: str
    ) -> "IncidentResolve":
        del root
        # save resolve status
        _, pk = pk_from_global_id(incident_id)
        incident = Incident.objects.get(pk=pk)

        if not has_zone_permission(
            info.context.user, incident.zone, INCIDENTS_RESOLVE
        ):
            raise PermissionDenied(
                "You do not have permission to resolve this incident."
            )

        # early termination if race conditions on mutation
        if incident.status == Incident.Status.RESOLVED:
            logger.warning(
                f"Attempting to resolve an incident which is already resolved (PK: {pk})"
            )
            return IncidentResolve(incident=incident)

        incident.status = Incident.Status.RESOLVED
        incident.save()

        # post save events
        incident_resolve = IncidentResolve(incident=incident)
        log_action(info.context.user, incident, Comment.ActivityType.RESOLVE)
        incident_resolve.send_email(info, incident)

        return incident_resolve


class IncidentReopen(BaseMutation):
    class Arguments:
        incident_id = graphene.ID(required=True)

    incident = graphene.Field(IncidentType)

    @staticmethod
    def mutate(
        root: None, info: graphene.ResolveInfo, incident_id: str
    ) -> "IncidentReopen":
        del root
        # save reopen status
        _, pk = pk_from_global_id(incident_id)
        incident = Incident.objects.get(pk=pk)

        # early termination if race conditions on mutation
        if incident.status == Incident.Status.OPEN:
            logger.warning(
                f"Attempting to open an incident which is already open (PK: {pk})"
            )
            return IncidentReopen(incident=incident)

        incident.status = Incident.Status.OPEN
        incident.save()

        # post save events
        log_action(info.context.user, incident, Comment.ActivityType.REOPEN)

        return IncidentReopen(incident=incident)


class IncidentExportVideo(BaseMutation):
    class Arguments:
        incident_id = graphene.ID(required=True)
        labeled = graphene.Boolean()

    download_url = graphene.String()

    @staticmethod
    def get_signed_url(incident: Incident, labeled: bool) -> str:
        """Helper method to get signed url from incident data and data storage provider

        Args:
            incident (Incident): an instance of the incident
            labeled (bool): whether video is labeled or not

        Raises:
            ValueError: if no video path
            ValueError: if no annotations found

        Returns:
            str: signed url
        """
        provider = None
        incident_video_path = None

        if incident.video_s3_path:
            incident_video_path = incident.video_s3_path
            incident_annotations_path = incident.annotations_s3_path
            annotated_path = incident.video_annotated_s3_path
            annotated_path_prefix = "video_annotated_s3_path"
            bucket_prefix = "s3://"
            provider = StorageProvider.S3

        if not incident_video_path:
            raise ValueError(
                "No video_path found for incident",
                f"(PK: {incident.pk}, UUID: {incident.uuid}) provider: {provider}",
            )

        download_path: Optional[str] = None
        download_filename: Optional[str] = None

        if labeled:
            if not incident_annotations_path:
                raise ValueError(
                    "No annotations_path found for incident",
                    f"(PK: {incident.pk}, UUID: {incident.uuid}) provider: {provider}",
                )

            download_filename = (
                f"voxel_incident_with_labels_{incident.uuid}.mp4"
            )

            if annotated_path:
                download_path = annotated_path
            else:
                # Combine video and annotation as new video and store to Google Cloud
                base_relative_path = "/".join(
                    incident_video_path[len(bucket_prefix) :].split("/")[:-1]
                )
                video_annotated_filename = f"{incident.uuid}_annotated.mp4"
                video_annotated_path = os.path.join(
                    f"{bucket_prefix}{base_relative_path}",
                    video_annotated_filename,
                )

                # TODO: perform rendering in background job
                video_rendering(
                    incident_annotations_path,
                    incident_video_path,
                    video_annotated_path,
                    incident.actor_ids,
                    provider,
                )

                incident.data = {
                    **incident.data,
                    annotated_path_prefix: video_annotated_path,
                }
                incident.save()

                download_path = video_annotated_path
        else:
            download_filename = f"voxel_incident_{incident.uuid}.mp4"
            download_path = incident_video_path

        if provider == StorageProvider.S3:
            bucket, path = aws_utils.separate_bucket_from_relative_path(
                download_path
            )
            signed_url = aws_utils.generate_presigned_url(
                bucket,
                path,
                response_disposition=attachment_disposition(download_filename),
            )

        return signed_url

    @staticmethod
    def mutate(
        root: None,
        info: graphene.ResolveInfo,
        incident_id: str,
        labeled: bool,
    ) -> "IncidentExportVideo":
        """Mutation function to export a video for download
        Args:
            root (None): root object
            info (graphene.ResolveInfo): info object that contains context
            incident_id (str): id of requested incident
            labeled (bool): whether we want the expor to be labeled or not
        Raises:
            ValueError: if no path
            ValueError: if annotations s3 path not found
        Returns:
            IncidentExportVideoV2: returning the download URL
        """
        del root, info
        # trunk-ignore(pylint/C0103): pk is ok to use
        _, pk = pk_from_global_id(incident_id)
        incident = Incident.objects_raw.get(pk=pk)

        signed_url = IncidentExportVideo.get_signed_url(
            incident=incident,
            labeled=labeled,
        )

        return IncidentExportVideo(download_url=signed_url)


class IncidentCreateShareLink(BaseMutation):
    """Ability for a user to share an incident for viewing."""

    class Arguments:
        incident_id = graphene.ID(required=True)

    share_link = graphene.String()

    @staticmethod
    def mutate(
        root: None,
        info: graphene.ResolveInfo,
        incident_id: str,
    ) -> "IncidentCreateShareLink":
        """Mutation to create a shareable link for an incident.

        Args:
            root (None): root object
            info (graphene.ResolveInfo): info object that contains context
            incident_id (str): id of requested incident

        Raises:
            PermissionDenied: if user does not have the proper permissions to generate share link

        Returns:
            IncidentCreateShareLink: returning the shareable url
        """
        del root

        _, incident_pk = pk_from_global_id(incident_id)
        incident = Incident.objects.get(pk=incident_pk)

        if not has_zone_permission(
            info.context.user,
            incident.zone,
            INCIDENT_PUBLIC_LINK_CREATE,
        ):
            raise PermissionDenied(
                "You do not have permission to create share links for users in this zone."
            )

        share_link = ShareLink.generate(
            shared_by=info.context.user,
            incident=incident,
        )

        return IncidentCreateShareLink(
            share_link=share_link,
        )


class IncidentHighlight(BaseMutation):
    class Arguments:
        incident_id = graphene.ID(required=True)

    incident = graphene.Field(IncidentType)

    @staticmethod
    def mutate(
        root: None,
        info: graphene.ResolveInfo,
        incident_id: str,
    ) -> "IncidentHighlight":
        """Mark incident as highlighted.

        Args:
            root (None): root graphene object
            info (graphene.ResolveInfo): graphene context
            incident_id (str): incident ID

        Raises:
            PermissionDenied: when user has insufficient permissions

        Returns:
            IncidentHighlight: updated incident
        """
        del root
        _, incident_pk = pk_from_global_id(incident_id)

        if not has_global_permission(info.context.user, INCIDENTS_HIGHLIGHT):
            raise PermissionDenied(
                "You are not allowed to modify the highlighted flag for this incident."
            )

        incident = Incident.objects.get(pk=incident_pk)
        incident.highlighted = True
        incident.save()
        return IncidentHighlight(incident=incident)


class IncidentUndoHighlight(BaseMutation):
    class Arguments:
        incident_id = graphene.ID(required=True)

    incident = graphene.Field(IncidentType)

    @staticmethod
    def mutate(
        root: None,
        info: graphene.ResolveInfo,
        incident_id: str,
    ) -> "IncidentUndoHighlight":
        """Mark incident as NOT highlighted.

        Args:
            root (None): root graphene object
            info (graphene.ResolveInfo): graphene context
            incident_id (str): incident ID

        Raises:
            PermissionDenied: when user has insufficient permissions

        Returns:
            IncidentHighlight: updated incident
        """
        del root
        _, incident_pk = pk_from_global_id(incident_id)

        if not has_global_permission(info.context.user, INCIDENTS_HIGHLIGHT):
            raise PermissionDenied(
                "You are not allowed to modify the highlighted flag for this incident."
            )

        incident = Incident.objects.get(pk=incident_pk)
        incident.highlighted = False
        incident.save()
        return IncidentHighlight(incident=incident)
