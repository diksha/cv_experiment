import os
import sys

from notebook import notebookapp


def update_argv():
    """Update sys.argv with custom configuration values."""
    workspace_dir = os.environ["BUILD_WORKSPACE_DIRECTORY"]
    sys.argv.append(f"--notebook-dir={workspace_dir}")


if __name__ == "__main__":
    update_argv()
    sys.exit(notebookapp.main())
