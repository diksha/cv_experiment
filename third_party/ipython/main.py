import os
import sys

from IPython import start_ipython


def update_argv():
    """Update sys.argv with custom configuration values."""
    sys.path.append(os.environ["BUILD_WORKSPACE_DIRECTORY"])


if __name__ == "__main__":
    update_argv()
    sys.exit(start_ipython())
