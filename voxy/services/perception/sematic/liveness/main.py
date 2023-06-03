"""
This is the entry point of your pipeline.

This is where you import the pipeline function from its module and resolve it.
"""
import argparse

from core.infra.sematic.shared.utils import (
    PipelineSetup,
    SematicOptions,
    resolve_sematic_future,
)
from services.perception.sematic.liveness.pipeline import pipeline


def main(sematic_options: SematicOptions):
    """
    Entry point of my pipeline.

    Args:
      sematic_options (SematicOptions): options for Sematic resolvers

    Raises:
      RuntimeError: invalid result from sematic function
    """
    # trunk-ignore(pylint/E1101)
    future = pipeline(pipeline_setup=PipelineSetup()).set(
        name="Sematic liveness check", tags=["P0"]
    )
    resolve_sematic_future(future, sematic_options, block_run=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    SematicOptions.add_to_parser(parser)
    args = parser.parse_args()
    main(SematicOptions.from_args(args))
