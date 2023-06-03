import typing as t

from django.utils import timezone as tzone
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.exceptions import ValidationError
from rest_framework.generics import ListCreateAPIView
from rest_framework.pagination import LimitOffsetPagination
from rest_framework.request import Request
from rest_framework.response import Response

from core.portal.api.models.share_link import ShareLink
from core.portal.incidents.commands import IngestIncident
from core.structs.incident import Incident as IncidentStruct


class IncidentsEndpoint(ListCreateAPIView):
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = LimitOffsetPagination

    def get_queryset(self, *_: str, **__: str) -> None:
        raise NotImplementedError(
            "Please do not use this endpoint, use /graphql instead"
        )

    def post(self, request: Request, *args, **kwargs) -> Response:
        """Creates an incident from the POST request data.

        Args:
            request (Request): request
            args: unused args
            kwargs: unused kwargs

        Returns:
            Response: response
        """
        del args, kwargs

        data = self._get_request_data_as_dict(request)
        incident_struct = IncidentStruct.from_dict(data)
        incident = IngestIncident(incident_struct).execute()
        return Response({"id": incident.pk})

    def _get_request_data_as_dict(
        self,
        request: Request,
    ) -> t.Dict[str, t.Any]:
        """Converts the request data to a dict.

        Args:
            request (Request): request

        Returns:
            t.Dict: request data dict
        """
        data = {}
        for key in request.data.keys():
            # HACK: patch list fields since QueryDict.get() is
            #       stupid and returns the last value in the list
            # TODO: remove this hack once request.data is POSTed as JSON
            treat_as_list = key == "tail_incident_uuids" and hasattr(
                request.data, "getlist"
            )

            if treat_as_list:
                data[key] = request.data.getlist(key)
            else:
                data[key] = request.data.get(key)

        return data


@api_view(["GET"])
@permission_classes([permissions.AllowAny])
def share_incident(request: Request, **kwargs: str) -> Response:
    """Incident share link redemption.

    :param request: request sent to redeem share link
    :param kwargs: key word arguments
    :raises ValidationError: if non-relevant token
    :returns: Response.
    """
    share_link = ShareLink.objects.get(token=kwargs["token"])

    if share_link.expires_at < tzone.now():
        raise ValidationError("Token has expired.")

    share_link.increment_visits()
    incident = share_link.incident

    return Response(
        dict(
            id=incident.id,
            zone_name=incident.zone.name,
            title=incident.title,
            video_url=incident.video_url,
            camera_name=incident.camera.name,
            timestamp=incident.timestamp,
            annotations_url=incident.annotations_url,
            actor_ids=incident.actor_ids,
        )
    )
