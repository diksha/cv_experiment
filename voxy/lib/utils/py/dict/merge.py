from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

DEFAULT_KEY_DELIMITER = "."


class ConflictStrategy(Enum):
    OVERRIDE = 1
    DO_NOT_UPDATE = 2
    THROW_ERROR = 3


class KeyConflictError(Exception):
    def __init__(self, key, src_defined):
        self.message = f"Key '{key}' already defined by {src_defined}"
        super().__init__(self.message)


class DictionaryMerger:
    """DictionaryMerger provides features for combining configs together"""

    def __init__(self, key_delimiter: str = DEFAULT_KEY_DELIMITER):
        """Initialize a DictionaryMerger

        Args:
            key_delimiter (str, optional): Delimiter for nested keys.
            Keys in source dicts must not contain the delimiter.
            Defaults to "."
        """
        self._key_delimiter = key_delimiter
        self._sources: Dict[str, List[str]] = {}
        self._flat_config: Dict[str, any] = {}

    def _update(self, key, value, source):
        self._flat_config[key] = value
        self._sources.setdefault(key, [])
        self._sources[key].append(source)

    def apply(
        self,
        config: dict,
        src: str,
        strategy: ConflictStrategy = ConflictStrategy.OVERRIDE,
    ):
        """Update the config with a new configuration

        Args:
            config (dict): Additional config to apply to the current configuration
            src (str, optional):
                Used for tracking where each config value comes from.
            strategy (ConflictStrategy):
                Determines how to deal with conflicts in the config values
                Defaults to OVERRIDE

        Raises:
            ValueError:
                if keys in the config are formatted invalidly
                if invalid ConfictStrategy is given
            KeyConflictError:
                if choosing to throw error on the occurence of conflicting config values
        """
        _validate_keys(config, self._key_delimiter)

        for key, val in _flatten_config(config, self._key_delimiter).items():
            if key not in self._flat_config:
                self._update(key, val, src)
            elif strategy is ConflictStrategy.OVERRIDE:
                self._update(key, val, src)
            elif strategy is ConflictStrategy.DO_NOT_UPDATE:
                # Add to front of sources since value in back represents current value
                self._sources[key] = [src] + self._sources[key]
            elif strategy is ConflictStrategy.THROW_ERROR:
                raise KeyConflictError(key, self._sources[key][-1])
            else:
                raise ValueError(f"Unsupported ConflictStrategy: {strategy}")

    def get_merged(self) -> dict:
        """
        Returns:
            dict: the combined result of all applied configs
        """
        return _deepen_config(self._flat_config, self._key_delimiter)

    def get_definition_source(
        self, key: Union[List[str], str]
    ) -> Optional[str]:
        """
        Args:
            key (Union[List[str], str]): The path of keys to a nested value
                Either as a list of keys or a string of keys which is delimited by "."

        Returns:
            Optional[str]: The source that defines the value at 'key'
                None if the key does not exist
        """
        sources = self.get_definition_sources(key)
        return sources[-1] if len(sources) != 0 else None

    def get_definition_sources(self, key: Union[List[str], str]) -> List[str]:
        """
        Args:
            key (Union[List[str], str]): The path of keys to a nested value
                Either as a list of keys or a string of keys which is delimited by "."

        Returns:
            List[str]: All the sources that have a definition for the key.
                The last element will be the defining source
        """
        if isinstance(key, list):
            key = self._key_delimiter.join(key)

        return self._sources.get(key, [])

    def get_definition_sources_dict(self) -> Dict[str, Tuple[str, List[str]]]:
        """
        Returns:
            Dict[str, Tuple[str, List[str]]]:
                A dictionary mapping keys to a tuple of the defining source and all sources
                that have a definition for the key
        """
        return {
            key: (sources[-1], sources)
            for key, sources in self._sources.items()
        }


def _key_is_valid(key: str, key_delimiter: str) -> bool:
    """Keys must be string and must not include the delimeter used for
    denoting the key path for a nested value

    Args:
        key (str): key in a config dict

    Returns:
        bool: true if the key satisfies constraints
    """
    return isinstance(key, str) and (key_delimiter not in key)


def _do_validate_keys(config: dict, key_delimiter: str) -> str:
    """Returns the first key in the config that is improperly formatted

    Args:
        config (dict): nested dictionary
        key_delimiter (str): delimiter used for denoting the key path for a nested value

    Returns:
        Optional[str]: invalid key found in the key, None if all keys are valid
    """
    for key, value in config.items():
        if not _key_is_valid(key, key_delimiter):
            return key

        if isinstance(value, dict):
            invalid_key = _do_validate_keys(value, key_delimiter)
            if invalid_key is not None:
                return invalid_key

    return None


def _validate_keys(config: dict, key_delimiter: str):
    """Ensures keys are properly formatted, i.e does not contain the '.' character

    Args:
        config (dict): nested config
        key_delimiter (str): delimiter used for denoting the key path for a nested value

    Raises:
        ValueError: on being passed a config with an invalid key
    """
    invalid_key = _do_validate_keys(config, key_delimiter)
    if invalid_key is not None:
        raise ValueError(
            f"key '{invalid_key}' in config is improperly formatted"
        )


def _flatten_config(config: dict, key_delimiter: str, parent_key="") -> dict:
    """Flattens a dictionary

    Args:
        config (dict): config to flatten
        parent_key (str, optional): Used for tracking key path in recursion. Defaults to "".

    Returns:
        dict: flattened version of config with keys delimited by key_delimiter
    """
    items = []
    for key, val in config.items():
        new_key = parent_key + key_delimiter + key if parent_key else key
        if isinstance(val, dict):
            items.extend(_flatten_config(val, key_delimiter, new_key).items())
        else:
            items.append((new_key, val))
    return dict(items)


def _deepen_config(flat_config: dict, key_delimiter: str) -> dict:
    """
    Takes a one-dimensional dictionary with and deepens it
    by separating keys by key_delimiter

    Args:
        flat_config (dict): config with no nested values

    Returns:
        dict: config with nested values
    """
    deep_config = {}
    for key, value in flat_config.items():
        keys = key.split(key_delimiter)
        cur_config = deep_config
        for new_key in keys[:-1]:
            cur_config.setdefault(new_key, {})
            cur_config = cur_config[new_key]
        cur_config[keys[-1]] = value
    return deep_config
