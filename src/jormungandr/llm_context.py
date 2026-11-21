"""Actor-owned, immutable experience context for optional language policies.

Memory survives episode boundaries only when the actor explicitly retains this
object. A fresh trial/split/agent gets a fresh context; the learner has no shared
conversation cache. Snapshots in observations are the inputs replayed by PPO.
"""

from dataclasses import asdict, dataclass, replace
import math


@dataclass(frozen=True)
class Experience:
    episode_id: str
    timestep: int
    observation: str
    action: str
    outcome: str
    reward: float
    policy_version: int

    def __post_init__(self):
        if not self.episode_id or self.timestep < 0 or self.policy_version < 0:
            raise ValueError("experience requires episode, timestep and policy version")
        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite")


@dataclass(frozen=True)
class AgentContext:
    actor_id: str
    trial_id: str
    split: str
    max_events: int = 16
    events: tuple[Experience, ...] = ()

    def __post_init__(self):
        if not self.actor_id or not self.trial_id:
            raise ValueError("actor_id and trial_id are required")
        if self.split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if self.max_events < 0 or len(self.events) > self.max_events:
            raise ValueError("invalid context event limit")

    def append(self, event: Experience) -> "AgentContext":
        if self.events:
            last = self.events[-1]
            if last.episode_id == event.episode_id and event.timestep <= last.timestep:
                raise ValueError("experience must advance within an episode")
        events = (self.events + (event,))[-self.max_events :] if self.max_events else ()
        return replace(self, events=events)

    def metadata(self, observation: str, descriptions: dict[str, str]) -> dict:
        """Snapshot before choosing an action; append its outcome afterwards."""
        return {
            "llm": {
                "schema": "jormungandr.llm_context.v1",
                "actor_id": self.actor_id,
                "trial_id": self.trial_id,
                "split": self.split,
                "observation": observation,
                "candidate_descriptions": dict(descriptions),
                "history": [asdict(event) for event in self.events],
            }
        }
