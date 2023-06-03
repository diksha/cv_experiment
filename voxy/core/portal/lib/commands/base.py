import typing as t
from abc import ABC, abstractmethod


class CommandABC(ABC):
    """Abstract base class for commands."""

    @abstractmethod
    def execute(self) -> t.Any:
        """Execute the command.

        Returns:
            t.Any: the result of the command
        """
        raise NotImplementedError(
            "All command classes must implement the .execute() method"
        )
