import hashlib
import re

_MODEL_NAME_SALT = "_do_not_manually_generate_triton_model_names_"


def triton_model_name(artifact_model_path: str, is_ensemble: bool) -> str:
    """
    The triton model name is a hash of the artifact model path and a salt.

    Args:
        artifact_model_path (str): the artifact model path
        is_ensemble (str): whether or not the model is an ensemble

    Returns:
        str: the triton model name as it would exist in the model repo
    """
    name_hash = hashlib.sha256()
    name_hash.update(artifact_model_path.encode())
    name_hash.update(_MODEL_NAME_SALT.encode())
    suffix = name_hash.hexdigest()[:6]
    if is_ensemble:
        suffix = f"{suffix}_ensemble"

    sanitized_model_path = re.sub(r"[^a-zA-Z0-9_]", "_", artifact_model_path)

    return f"{sanitized_model_path}_{suffix}"
