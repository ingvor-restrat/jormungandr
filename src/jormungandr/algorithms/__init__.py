"""Built-in and externally discoverable Jörmungandr algorithms."""

from .base import (
    ActionResult,
    AlgorithmPlugin,
    BuildStructuredAgent,
    StructuredLearnerAgent,
    UpdateResult,
    normalize_update_result,
)
from .registry import (
    ENTRY_POINT_GROUP,
    algorithm_registry,
    canonical_algorithm_name,
)

# Registration is intentionally module-based: every built-in is replaceable by
# an entry-point plugin without teaching the service about its objective.
from . import appo as _appo  # noqa: F401,E402
from . import bc as _bc  # noqa: F401,E402
from . import c51 as _c51  # noqa: F401,E402
from . import cql as _cql  # noqa: F401,E402
from . import dqn as _dqn  # noqa: F401,E402
from . import dreamerv3 as _dreamerv3  # noqa: F401,E402
from . import impala as _impala  # noqa: F401,E402
from . import marwil as _marwil  # noqa: F401,E402
from . import maxent as _maxent  # noqa: F401,E402
from . import ppo as _ppo  # noqa: F401,E402
from . import qrdqn as _qrdqn  # noqa: F401,E402
from . import sac as _sac  # noqa: F401,E402
from . import structured_dqn as _structured_dqn  # noqa: F401,E402
from . import structured_bc as _structured_bc  # noqa: F401,E402
from . import structured_ppo as _structured_ppo  # noqa: F401,E402


def available_algorithms() -> list[str]:
    return algorithm_registry.names()


__all__ = [
    "ActionResult",
    "AlgorithmPlugin",
    "BuildStructuredAgent",
    "ENTRY_POINT_GROUP",
    "StructuredLearnerAgent",
    "StructuredDQNAgent",
    "StructuredPPOAgent",
    "StructuredPPOUpdate",
    "StructuredTransition",
    "UpdateResult",
    "algorithm_registry",
    "available_algorithms",
    "canonical_algorithm_name",
    "normalize_update_result",
]


def __getattr__(name: str):
    if name == "StructuredDQNAgent":
        from . import structured_dqn

        return getattr(structured_dqn, name)
    if name in {
        "StructuredPPOAgent",
        "StructuredPPOUpdate",
        "StructuredTransition",
    }:
        from . import structured_ppo

        return getattr(structured_ppo, name)
    raise AttributeError(name)
