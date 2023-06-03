import typing as t

from fakeredis import FakeStrictRedis


class CustomFakeStrictRedis(FakeStrictRedis):
    def get_many(self, keys: t.Iterable[str]) -> t.Dict[str, t.Any]:
        """Get multiple keys from the cache.

        Args:
            keys (Iterable[str]): An iterable containing the keys to be retrieved.

        Returns:
            t.Dict[str, t.Any]: A dictionary of key-value pairs
        """
        return {key: self.get(key) for key in keys}

    def delete_many(self, keys: t.Iterable[str] = ()) -> int:
        """Delete multiple keys from the cache.

        Args:
            keys (Iterable[str], optional): An iterable containing the keys to be deleted.

        Returns:
            int: The number of keys deleted
        """
        return sum(self.delete(key) for key in keys)

    def iter_keys(self, pattern: str = "*") -> t.Iterable[str]:
        """Iterate over keys in the cache that match the specified pattern.

        Args:
            pattern (str, optional): A pattern to match keys against. Defaults to '*'.

        Yields:
            Iterator[Iterable[str]]: An iterable of matching keys
        """
        for key in self.keys(pattern):
            yield key.decode("utf-8")

    def set(self, key: str, value: t.Any, *args, **kwargs) -> bool:
        """Set a value in the cache with optional timeout.

        Args:
            key (str): The key for the cache entry
            value: the value to store
            *args: extra input arguments
            **kwargs: extra key word arguments

        Returns:
            bool: True if the operation was successful, False otherwise
        """
        timeout = kwargs.pop("timeout", None)
        if timeout is not None:
            kwargs["ex"] = timeout
        return super().set(key, value, *args, **kwargs)

    def set_many(
        self, data: t.Dict[str, t.Any], timeout: t.Optional[int] = None
    ) -> None:
        """Set multiple keys in the cache.

        Args:
            data (t.Dict[str, t.Any]): key/value pairs to set in the cache
            timeout (t.Optional[int], optional): timeout in seconds
        """
        for key, value in data.items():
            self.set(key, value, timeout=timeout)

    def clear(self):
        """
        Clear all keys in the cache.
        """
        self.flushdb()
