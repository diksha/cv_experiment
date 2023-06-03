from typing import Optional, Set, Tuple, Union

import graphene


def pk_from_global_id(
    global_id: str,
    expected_type: Union[graphene.ObjectType, str] = None,
    raise_error: bool = False,
) -> Tuple[str, Optional[int]]:
    """Resolve Django primary key from GraphQL global ID."""
    try:
        _type, _pk = graphene.Node.from_global_id(global_id)
    except ValueError as resolution_error:
        raise RuntimeError(
            f"Couldn't resolve GraphQL ID: {global_id}."
        ) from resolution_error

    if expected_type and str(_type) != str(expected_type):
        if not raise_error:
            return _type, None
        raise RuntimeError(
            f"Expected {expected_type} ID, but received {str(_type)} ID."
        )
    return _type, _pk


def ids_to_primary_keys(id_values: Set[str]) -> Set[str]:
    """Converts a set of mixed IDs to a set of Django primary keys.
    Args:
        id_values (Set[str]): graphql global id
    Returns:
        Set[str]: A set of camera primary key
    """
    primary_key_set = set()
    for id_value in id_values:
        if id_value.isnumeric():
            # Assume numeric values are primary keys
            primary_key_set.add(id_value)
        else:
            _, primary_key = pk_from_global_id(id_value)
            primary_key_set.add(primary_key)
    return primary_key_set
