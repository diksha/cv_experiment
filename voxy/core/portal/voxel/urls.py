from django.contrib import admin
from django.urls import include, path
from django.views.decorators.csrf import csrf_exempt

from core.portal.accounts.endpoints import register
from core.portal.incidents.endpoints import IncidentsEndpoint, share_incident
from core.portal.lib.graphql.views import PrivateGrapheneView
from core.portal.lib.jobs.endpoints import hourly_trigger, semihourly_trigger

# trunk-ignore(pylint/W0611,flake8/F401)
from core.portal.visualizer.templates.dash import (
    plotly_apps,
    video_visualization,
)
from core.portal.voxel.endpoints import (
    commit_hash,
    liveness_check,
    readiness_check,
    trigger_error,
)

urlpatterns = [
    path("admin/", admin.site.urls),
    path(
        "api/",
        include(
            [
                path("incidents/", IncidentsEndpoint.as_view()),
                path(
                    "notifications/",
                    include(
                        [
                            path("trigger/hourly/", hourly_trigger),
                            path("trigger/semihourly/", semihourly_trigger),
                        ]
                    ),
                ),
                path("trigger_error/", trigger_error),
                path("register/<str:token>/", register),
                path("share/<str:token>/", share_incident),
                # TODO: delete /api/health once replaced by liveness/readiness
                path("health", liveness_check),
                path("commit_hash/", commit_hash),
                path(
                    "health/",
                    include(
                        [
                            path("liveness/", liveness_check),
                            path("readiness/", readiness_check),
                        ]
                    ),
                ),
            ]
        ),
    ),
    path(
        "internal/backend/django_plotly_dash/",
        include("django_plotly_dash.urls"),
    ),
    path("graphql/", csrf_exempt(PrivateGrapheneView.as_view(graphiql=True))),
]
