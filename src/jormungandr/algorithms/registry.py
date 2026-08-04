"""Algorithm plugin registry and Python entry-point discovery."""

from __future__ import annotations

from importlib import metadata
import re
import threading
from typing import Dict, Iterable

from .base import AlgorithmPlugin


ENTRY_POINT_GROUP = "jormungandr.algorithms"


def canonical_algorithm_name(value: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    aliases = {
        "maximum_entropy": "maxent",
        "maximum_entropy_rl": "maxent",
        "soft_q": "maxent",
        "dreamer_v3": "dreamerv3",
        "behavior_cloning": "bc",
        "qr_dqn": "qrdqn",
        "quantile_dqn": "qrdqn",
    }
    return aliases.get(name, name)


class AlgorithmRegistry:
    """Thread-safe registry that supports built-ins and installed packages."""

    def __init__(self) -> None:
        self._plugins: Dict[str, AlgorithmPlugin] = {}
        self._canonical: Dict[str, str] = {}
        self._entry_points_loaded = False
        self._lock = threading.RLock()

    def register(self, plugin: AlgorithmPlugin, *, replace: bool = False) -> None:
        name = canonical_algorithm_name(plugin.name)
        if not name:
            raise ValueError("algorithm plugin name is required")
        if plugin.default_export_module not in {"q", "policy", "heads"}:
            raise ValueError("default_export_module must be q, policy, or heads")
        if plugin.replay_mode not in {"transition", "trajectory", "supervision"}:
            raise ValueError(
                "replay_mode must be transition, trajectory, or supervision"
            )
        supported_representations = {"vector_discrete", "entity_candidates"}
        if (
            not plugin.representation_modes
            or len(set(plugin.representation_modes)) != len(plugin.representation_modes)
            or not set(plugin.representation_modes).issubset(supported_representations)
        ):
            raise ValueError(
                "representation_modes must contain unique supported modes"
            )
        if (
            "vector_discrete" in plugin.representation_modes
            and plugin.build is None
        ):
            raise ValueError("vector_discrete plugins require build")
        if (
            "entity_candidates" in plugin.representation_modes
            and plugin.build_structured is None
        ):
            raise ValueError(
                "entity_candidates plugins require build_structured"
            )
        keys = {name, *(canonical_algorithm_name(x) for x in plugin.aliases)}
        with self._lock:
            if name in self._plugins and not replace:
                raise ValueError(f"algorithm plugin already registered: {name}")
            self._plugins[name] = plugin
            for key in keys:
                if key:
                    self._canonical[key] = name

    def _load_entry_points(self) -> None:
        with self._lock:
            if self._entry_points_loaded:
                return
            self._entry_points_loaded = True
        try:
            discovered = metadata.entry_points()
            entries = (
                discovered.select(group=ENTRY_POINT_GROUP)
                if hasattr(discovered, "select")
                else discovered.get(ENTRY_POINT_GROUP, ())
            )
        except Exception:
            return
        for entry in entries:
            try:
                loaded = entry.load()
                plugin = loaded() if callable(loaded) and not isinstance(loaded, AlgorithmPlugin) else loaded
                if not isinstance(plugin, AlgorithmPlugin):
                    continue
                self.register(plugin)
            except Exception:
                # One broken optional package must not make the built-in runtime unusable.
                continue

    def get(self, name: str) -> AlgorithmPlugin:
        self._load_entry_points()
        key = canonical_algorithm_name(name)
        with self._lock:
            canonical = self._canonical.get(key, key)
            plugin = self._plugins.get(canonical)
        if plugin is None:
            supported = ", ".join(self.names())
            raise KeyError(f"unsupported algorithm plugin: {name}; available: {supported}")
        return plugin

    def names(self) -> list[str]:
        self._load_entry_points()
        with self._lock:
            return sorted(self._plugins)

    def plugins(self) -> Iterable[AlgorithmPlugin]:
        self._load_entry_points()
        with self._lock:
            return tuple(self._plugins[name] for name in sorted(self._plugins))


algorithm_registry = AlgorithmRegistry()
