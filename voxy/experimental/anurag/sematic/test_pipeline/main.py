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
from experimental.anurag.sematic.test_pipeline.pipeline import pipeline


def main(sematic_options: SematicOptions):
    """
    Entry point of my pipeline.

    Args:
      sematic_options (SematicOptions): options for Sematic resolvers
    """
    # trunk-ignore(pylint/E1101)
    future = pipeline(pipeline_setup=PipelineSetup()).set(
        name="Anurag's test pipeline"
    )
    resolve_sematic_future(future, sematic_options)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, required=True)
    SematicOptions.add_to_parser(parser)
    args = parser.parse_args()
    print(f"Test argument is : {args.test}")
    main(SematicOptions.from_args(args))
