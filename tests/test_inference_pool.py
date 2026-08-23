from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch

from jormungandr.inference_pool import InferencePoolConfig, ReplicatedInferencePool
from jormungandr.service import JormungandrRuntime


class ToyAgent:
    def __init__(self, device: str, state) -> None:
        self.device = torch.device(device)
        self.model = torch.nn.Linear(1, 1, bias=False).to(self.device)
        self.load_state_dict(state)

    def state_dict(self):
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state) -> None:
        self.model.load_state_dict(state["model"])


def test_pool_batches_requests_and_activates_one_version_across_replicas() -> None:
    source = ToyAgent("cpu", {"model": {"weight": torch.tensor([[1.0]])}})

    def factory(device, state):
        return ToyAgent(device, state)

    def infer(agent, payloads, metadata):
        values = torch.tensor(payloads, dtype=torch.float32, device=agent.device).reshape(-1, 1)
        with torch.no_grad():
            return (agent.model(values).reshape(-1) + float(metadata["offset"])).tolist()

    pool = ReplicatedInferencePool(
        InferencePoolConfig(
            devices=("cpu",),
            replicas=2,
            max_batch_size=4,
            batch_wait_ms=1.0,
        ),
        agent_factory=factory,
        infer_batch=infer,
    )
    pool.start(source.state_dict(), policy_version=0, metadata={"offset": 0.0})
    try:
        with ThreadPoolExecutor(max_workers=16) as executor:
            first = list(executor.map(lambda value: pool.submit(value), range(32)))
        assert [result.value for result in first] == [float(value) for value in range(32)]
        before = pool.snapshot()
        assert before["replica_count"] == 2
        assert sum(row["items"] for row in before["replicas"]) == 32

        source.model.weight.data.fill_(3.0)
        pool.refresh(source.state_dict(), policy_version=7, metadata={"offset": 1.0})
        after = pool.snapshot()
        assert {row["policy_version"] for row in after["replicas"]} == {7}
        assert len({row["activated_ts"] for row in after["replicas"]}) == 1
        refreshed = pool.submit_many([2.0, 4.0])
        assert [result.value for result in refreshed] == [7.0, 13.0]
        assert {result.policy_version for result in refreshed} == {7}
    finally:
        pool.close()


def test_vector_runtime_routes_policy_calls_through_replicas() -> None:
    runtime = JormungandrRuntime(
        inference_devices=["cpu"],
        inference_replicas=2,
        inference_max_batch_size=8,
        inference_batch_wait_ms=1.0,
    )
    runtime.create_model(
        obs_dim=2,
        model_id="replicated-policy",
        tensorboard_enabled=False,
        learner={
            "enabled": True,
            "device": "cpu",
            "hidden": 8,
            "min_replay": 1000,
            "checkpoint_every": 0,
        },
    )
    try:
        with ThreadPoolExecutor(max_workers=16) as executor:
            rows = list(
                executor.map(
                    lambda value: runtime.policy_infer(
                        "replicated-policy", obs=[float(value), 0.0]
                    ),
                    range(32),
                )
            )
        assert {row["inference_device"] for row in rows} == {"cpu"}
        assert {row["inference_replica_id"] for row in rows}.issubset({0, 1})
        pool = runtime.get_model("replicated-policy")["inference"]["pool"]
        assert pool["replica_count"] == 2
        assert sum(row["items"] for row in pool["replicas"]) == 32
        assert pool["activation"] == "double_buffered_atomic"
    finally:
        runtime.close_all()
