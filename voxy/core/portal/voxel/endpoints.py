import os

from django.db import connections
from django.db.migrations.executor import MigrationExecutor
from rest_framework import permissions, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.request import Request
from rest_framework.response import Response


@api_view(["GET"])
@permission_classes([permissions.AllowAny])
def readiness_check(request: Request) -> Response:
    """Readiness check.

    Args:
        request (Request): HTTP request

    Returns:
        Response: HTTP response (200 status if ready, otherwise 503)

    This check ensures all of the database migrations the currently running
    code depends on have been applied to the database.
    """
    del request
    for connection in connections.all():
        executor = MigrationExecutor(connection)
        plan = executor.migration_plan(executor.loader.graph.leaf_nodes())
        if plan:
            # The database does not have required migrations, return unhealthy
            return Response(status=status.HTTP_503_SERVICE_UNAVAILABLE)
    return Response(status=status.HTTP_200_OK)


@api_view(["GET"])
@permission_classes([permissions.AllowAny])
def liveness_check(request: Request) -> Response:
    """Liveness check.

    This check simply returns a 200 OK response, signifying the server is
    running and can receive requests.

    Args:
        request (Request): HTTP request

    Returns:
        Response: HTTP response with 200 status
    """
    del request
    return Response(status=status.HTTP_200_OK)


@api_view(["GET"])
@permission_classes([permissions.AllowAny])
def trigger_error(request: Request) -> Response:
    """Dummy function to trigger unhandled ZeroDevisionError exception.

    Args:
        request (Request): HTTP request

    Raises:
        ZeroDevisionError: every time (for testing purposes)

    Returns:
        Response: HTTP response with 50 status
    """
    del request
    1 / 0  # trunk-ignore(pylint/W0104)
    return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["GET"])
@permission_classes([permissions.AllowAny])
def commit_hash(request: Request) -> Response:
    """Function to return current git revision hash.

    Args:
        request (Request): HTTP request

    Returns:
        Response: HTTP response
    """
    del request

    git_revision = os.getenv("IMAGE_TAG", "unknown")

    return Response(git_revision)
