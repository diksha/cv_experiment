"""
This is the module in which you define your pipeline functions.

Feel free to break these definitions into as many files as you want for your
preferred code structure.
"""

import sematic

from core.infra.sematic.shared.resources import CPU_1CORE_4GB
from core.infra.sematic.shared.utils import PipelineSetup


@sematic.func(
    resource_requirements=CPU_1CORE_4GB,
    standalone=True,
)
def pipeline(pipeline_setup: PipelineSetup) -> str:
    """
    The root function of the pipeline.

    Args:
        pipeline_setup (PipelineSetup): setting up some defaults for pipeline

    Returns:
        int: dummy value
    """
    return "LIVE"
