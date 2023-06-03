import argparse
import os
import shutil
import signal

# trunk-ignore-all(bandit/B404)
# trunk-ignore-all(bandit/B603)
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime

import django
import psycopg2
from django.core import management
from loguru import logger

DOCKER_PATH = shutil.which("docker")
DBNAME = "voxeldev"
DBUSER = "voxelapp"
DBPASS = "voxelvoxel"
DBPORT = "31123"

CONTAINER_NAME = "voxel-polygon-schema-export-tmp"


@contextmanager
def docker_postgres_process():
    """Runs a docker postgres process with cleanup

    Yields:
        subproces.Popen: the process handler to the docker process
    """
    docker_process = subprocess.Popen(
        [
            DOCKER_PATH,
            "run",
            "--rm",
            "--name",
            CONTAINER_NAME,
            "-e",
            f"POSTGRES_PASSWORD={DBPASS}",
            "-e",
            f"POSTGRES_USER={DBUSER}",
            "-e",
            f"POSTGRES_DB={DBNAME}",
            "-p",
            f"127.0.0.1:{DBPORT}:5432",
            "postgres",
        ],
    )

    try:
        yield docker_process
    finally:
        docker_process.send_signal(signal.SIGINT)
        try:
            docker_process.wait(10.0)
        except subprocess.TimeoutExpired:
            docker_process.kill()


def wait_for_postgres():
    """Waits for the docker postgres process to be available, timing out after 60s

    Raises:
        RuntimeError: docker never became available
    """
    start = datetime.now()

    while (datetime.now() - start).seconds < 60.0:
        try:
            conn = psycopg2.connect(
                " ".join(
                    [
                        f"dbname='{DBNAME}'",
                        f"user='{DBUSER}'",
                        "host='127.0.0.1'",
                        f"password='{DBPASS}'",
                        f"port='{DBPORT}'",
                        "connect_timeout=1",
                    ]
                )
            )
            conn.close()
            return
        # trunk-ignore(pylint/W0703)
        except Exception as exc:
            logger.warning(
                f"Waiting for DB connection to become available: {exc}"
            )
        time.sleep(1.0)

    raise RuntimeError("docker postgress process never became available")


def main():
    """Runs the dump"""
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", help="file to write output schema to")
    args = parser.parse_args()

    with docker_postgres_process():
        wait_for_postgres()

        os.environ.setdefault(
            "DJANGO_SETTINGS_MODULE", "core.portal.voxel.settings"
        )
        os.environ.setdefault("ENVIRONMENT", "development")
        os.environ.setdefault("DEFAULT_DB_PORT", DBPORT)

        django.setup()
        management.call_command("migrate", "--database", "default")

        result = subprocess.run(
            [
                DOCKER_PATH,
                "exec",
                CONTAINER_NAME,
                "pg_dump",
                "-d",
                DBNAME,
                "-U",
                DBUSER,
                "--schema-only",
            ],
            capture_output=True,
            check=True,
        )

    with open(args.output_file, "wb+") as outf:
        outf.write(result.stdout)


if __name__ == "__main__":
    main()
