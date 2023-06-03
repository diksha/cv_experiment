import unittest

from core.utils.print_utils import TermColors

_SOURCE_SCHEMA = "core/portal/lib/graphql/schema.graphql"
_GENERATED_SCHEMA = "core/portal/lib/graphql/generated_schema.graphql"
_FAILURE_MESSAGE = f"""
{TermColors.WARNING}
Your schema.graphql file is out of sync with source files. Please run the
following script to regenerate the schema file and client code:
{TermColors.OKGREEN}
    $ ./services/portal/scripts/codegen.sh
{TermColors.ENDC}{TermColors.WARNING}
If you didn't intentionally modify the schema then please verify that
whatever changes you did make are not introducing breaking changes.
Even if you did intentionally modify the schema, please verify :)
{TermColors.ENDC}
"""


class SchemaSyncTest(unittest.TestCase):
    def test_schema_is_in_sync(self) -> None:
        """Ensures the GraphQL schema file is synced with source files."""

        with open(_SOURCE_SCHEMA, encoding="UTF-8") as source_file, open(
            _GENERATED_SCHEMA, encoding="UTF-8"
        ) as generated_file:
            source = source_file.readlines()
            generated = generated_file.readlines()
            self.assertEqual(source, generated, _FAILURE_MESSAGE)


if __name__ == "__main__":
    unittest.main()
