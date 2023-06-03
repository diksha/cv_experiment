import re
from typing import Any, Dict, List, Optional

from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpRequest
from graphene_django.views import GraphQLView as GrapheneView
from graphql import GraphQLError
from graphql.execution import ExecutionResult

COMMENT_PATTERN = r"#.*$"


class PrivateGrapheneView(LoginRequiredMixin, GrapheneView):
    raise_exception = True

    def execute_graphql_request(
        self,
        request: HttpRequest,
        data: Optional[Dict[str, Any]],
        query: Optional[str],
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ):
        """Overrides graphql execution

        Args:
            request (HttpRequest): incoming http request
            data (Optional[Dict[str, Any]]): optional data
            query (Optional[str]): optional query string
            *args: list of args
            **kwargs: dict of kwargs

        Returns:
            ExecutionResult: output of graphql execution
        """

        def trim_comments(string_to_trim: str) -> str:
            """Replace all comment lines with an empty string

            Args:
                string_to_trim (str): input string

            Returns:
                str: stripped string
            """
            return re.sub(
                COMMENT_PATTERN, "", string_to_trim, flags=re.MULTILINE
            ).strip()

        if not query or not trim_comments(query):
            return ExecutionResult(
                data=[], errors=[GraphQLError("Query is required")]
            )
        return super().execute_graphql_request(
            request, data, query, *args, **kwargs
        )
