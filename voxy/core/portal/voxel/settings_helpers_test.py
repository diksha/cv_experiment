import os
from unittest import mock

from core.portal.voxel.settings_helpers import get_allowed_hosts


def test_get_allowed_hosts_handles_empty_args() -> None:
    """Tests empty args."""
    actual = get_allowed_hosts()
    expected = []
    assert actual == expected


def test_get_allowed_hosts_handles_single_arg() -> None:
    """Tests single arg value."""
    actual = get_allowed_hosts("foo")
    expected = ["foo"]
    assert actual == expected


def test_get_allowed_hosts_handles_multiple_args() -> None:
    """Tests multiple arg values."""
    actual = get_allowed_hosts("foo", "bar", "baz")
    expected = ["foo", "bar", "baz"]
    assert actual == expected


def test_get_allowed_hosts_ignores_empty_strings() -> None:
    """Tests empty string arg values."""
    actual = get_allowed_hosts("foo", "", "bar")
    expected = ["foo", "bar"]
    assert actual == expected


@mock.patch.dict(os.environ, {"HOST_IP": "127.0.0.1"})
@mock.patch.dict(os.environ, {"POD_IP": "127.0.0.2"})
def test_get_allowed_hosts_reads_host_and_pod_ip_from_environment() -> None:
    """Tests expected host values are read from environment."""
    actual = get_allowed_hosts("foo")
    expected = ["foo", "127.0.0.2", "127.0.0.1"]
    assert actual == expected
