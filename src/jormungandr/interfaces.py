from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Protocol

JsonDict = Dict[str, Any]


class EpisodeClient(ABC):
    """Single-episode interface over an external stepper."""

    @abstractmethod
    def initialize(self, init_params: Optional[Mapping[str, Any]] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self, step_params: Optional[Mapping[str, Any]] = None) -> JsonDict:
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError

    def remaining_steps(self) -> Optional[int]:
        return None

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class EpisodeFactory(Protocol):
    """Creates new episode clients from a per-episode config payload."""

    def create(self, config: Mapping[str, Any]) -> EpisodeClient:
        ...


class CommandTransport(ABC):
    """Request/response transport for JSON command protocols."""

    @abstractmethod
    def call(self, request: Mapping[str, Any]) -> JsonDict:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
