"""
This is the module in which you define your pipeline functions.

Feel free to break these definitions into as many files as you want for your
preferred code structure.
"""
import os

import sematic

from core.infra.sematic.shared.resources import GPU_1CPU_4GB_8x
from core.infra.sematic.shared.utils import PipelineSetup


@sematic.func(
    resource_requirements=GPU_1CPU_4GB_8x,
    standalone=True,
)
def pipeline(pipeline_setup: PipelineSetup) -> int:
    """
    The root function of the pipeline.

    Args:
        pipeline_setup (PipelineSetup): setting up some defaults for pipeline

    Returns:
        int: dummy value
    """
    print("Hello.")
    print(os.getenv("AWS_CONFIG_FILE"))
    return 1
