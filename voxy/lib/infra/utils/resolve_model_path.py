import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import boto3
from loguru import logger
from rules_python.python.runfiles import runfiles

_VOXEL_STORAGE_BUCKET = "voxel-storage"
# trunk-ignore(bandit/B108): intentionally shared
_MODEL_CACHE_DIR = "/tmp/voxel/model-cache"

# trunk-ignore-all(semgrep): semgrep hates this file


def _model_artifact_meta(model_path: str) -> dict:
    with open(
        runfiles.Create().Rlocation(
            "voxel/lib/infra/utils/model_manifest.json"
        ),
        encoding="utf-8",
    ) as jsonf:
        artifacts = json.load(jsonf)

    return artifacts[model_path]


def _fetch_model(
    model_path: str, boto3_session: Optional[boto3.Session]
) -> str:
    model_cache_path = os.path.join(_MODEL_CACHE_DIR, model_path)

    logger.debug(f"fetching model: {model_path} from s3")

    if os.path.exists(model_cache_path):
        logger.debug(f"found model on disk at {model_cache_path}")
        return model_cache_path

    try:
        os.makedirs(_MODEL_CACHE_DIR)
    except FileExistsError:
        pass

    if boto3_session is None:
        boto3_session = boto3

    runf = runfiles.Create()

    artifact_name = Path(model_path).parts[0]
    artifact_meta_path = os.path.join(artifact_name, "meta.json")
    artifact_meta_abspath = runf.Rlocation(artifact_meta_path)

    with open(artifact_meta_abspath, encoding="utf-8") as metaf:
        artifact_meta = json.load(metaf)

    artifact_name = artifact_meta["name"]
    url = urlparse(artifact_meta["url"])

    s3client = boto3_session.client("s3")

    with tempfile.TemporaryFile() as tempf:
        logger.debug(f"fetching {url}")
        s3client.download_fileobj(
            url.netloc, url.path.removeprefix("/"), tempf
        )
        tempf.seek(0)

        with tarfile.open(fileobj=tempf, mode="r") as tarf:
            artifact_dir = os.path.join(_MODEL_CACHE_DIR, artifact_name)
            with tempfile.TemporaryDirectory(dir=_MODEL_CACHE_DIR) as tempdir:
                artifact_tempdir = os.path.join(tempdir, artifact_name)
                tarf.extractall(artifact_tempdir)

                # we use a rename here so that this operation is atomic
                # this is important because otherwise it would be possible
                # for one runner to read a partially extracted model from
                # another runner if they are booting at the same time
                try:
                    os.rename(artifact_tempdir, artifact_dir)
                except FileExistsError:
                    # only reraise the exception if the destination directory
                    # does not exist. rename can raise an OSError if two
                    # processes attempted to download and extract the same
                    # model archive at the same time
                    if not os.path.exists(model_cache_path):
                        logger.exception(
                            "Failed to move extracted artifact directory to final destination"
                        )
                        raise

    if not os.path.exists(model_cache_path):
        artifact_url = artifact_meta["url"]
        raise RuntimeError(
            f"Downloaded artifact {artifact_url} did not contain model {artifact_name}"
        )

    logger.debug(f"downloaded model {model_cache_path}")
    return model_cache_path


def resolve_model_path(
    model_path: str, boto3_session: Optional[boto3.Session] = None
) -> str:
    """Attempts to resolve a model path which might be an artifact path.
       Noop if model_path is already a valid path

    Args:
        model_path (str): model artifact path
        boto3_session (boto3.Session): optional boto3 session to use

    Returns:
        str: valid absolute model path
    """
    # guard against a double resolve
    if os.path.exists(model_path):
        return model_path

    artifact_path = runfiles.Create().Rlocation(model_path)
    if artifact_path is not None and os.path.exists(artifact_path):
        return artifact_path

    return _fetch_model(model_path, boto3_session)


def unresolve_model_path(model_path: str) -> str:
    """Attempts to unresolve a model path which might be an artifact path.
       Noop if model_path is already unresolved

    Args:
        model_path (str): valid absolute model artifact path

    Returns:
        str: original model path name
    """
    unresolved_name = model_path.replace(_MODEL_CACHE_DIR, "").lstrip("/")
    return unresolved_name


def resolve_all_model_paths(config: dict):
    """Attempts to resolve all model paths in a graph config by searching for keys with
       the string "model_path" in which have string values. Any models paths that cannot
       be resolved to real files will throw a runtime error.

    Args:
        config (dict): a graph config dict
    """
    for key, value in config.items():
        if isinstance(value, dict):
            resolve_all_model_paths(value)

        if (
            isinstance(key, str)
            and isinstance(value, str)
            and ("model_path" in key)
        ):
            config[key] = resolve_model_path(value)
