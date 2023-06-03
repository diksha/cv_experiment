from uuid import UUID


def is_valid_uuid(uuid_to_test: str, version: int = 4) -> bool:
    """Check if uuid_to_test is a valid UUID.

    Args:
        uuid_to_test (str): uuid to validate
        version (int): uuid version

    Returns:
        bool: `True` if uuid_to_test is a valid UUID, otherwise `False`.

    Examples:
    >>> is_valid_uuid('c9bf9e57-1685-4c89-bafb-ff5af830be8a')
    True
    >>> is_valid_uuid('c9bf9e58')
    False
    >>> is_valid_uuid(1453)
    False
    """

    try:
        uuid_obj = UUID(uuid_to_test, version=version)
    except ValueError:
        return False
    return str(uuid_obj) == uuid_to_test
