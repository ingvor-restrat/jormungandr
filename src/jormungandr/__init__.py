"""Reusable Jormungandr service runtime (generic trainer/checkpoint/inference over HTTP)."""

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "JormungandrRuntime",
    "JormungandrHttpServer",
    "export_torchscript_from_checkpoint",
    "export_inference_bundle",
    "inspect_checkpoint",
    "compare_checkpoints",
]


def __getattr__(name: str) -> Any:
    if name in {"JormungandrRuntime", "JormungandrHttpServer"}:
        module = import_module("jormungandr.service")
        return getattr(module, name)
    if name in {
        "compare_checkpoints",
        "export_inference_bundle",
        "export_torchscript_from_checkpoint",
        "inspect_checkpoint",
    }:
        module = import_module("jormungandr.export")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
