"""Train and query a stock/option policy through Jormungandr's HTTP API.

Build the worker from the Volt checkout:
    cargo build -p volt-compute --release --bin native_learner

In an installed Jormungandr environment, start its service with the worker:
    export VOLT_LEARNER_BIN=/absolute/path/to/native_learner
    jormungandr-service --host 127.0.0.1 --port 8080

Run this client from the Jormungandr checkout in that environment:
    python examples/native_hedging_api.py --url http://127.0.0.1:8080

Scenario prices are synthetic. Quantities and prices use one underlying unit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import uuid

from jormungandr.client import JormungandrClient


def call_price(spot: float, maturity: float) -> float:
    sigma, rate, dividend = 0.2, 0.03, 0.01
    d1 = (math.log(spot / 100) + (rate - dividend + sigma**2 / 2) * maturity) / (sigma * math.sqrt(maturity))
    d2 = d1 - sigma * math.sqrt(maturity)
    normal = lambda x: (1 + math.erf(x / math.sqrt(2))) / 2
    return spot * math.exp(-dividend * maturity) * normal(d1) - 100 * math.exp(-rate * maturity) * normal(d2)


def scenario_batch(paths: int, first_path: int, seed: int) -> dict:
    rng = random.Random(seed)
    batch = {"paths": paths, "provenance": {"dataset_id": "gbm-study", "first_path": first_path},
             "marks": [], "features": [], "payoffs": []}
    for _ in range(paths):
        spots = [100.0]
        for _ in range(2):
            spots.append(spots[-1] * math.exp((0.03 - 0.01 - 0.2**2 / 2) / 252 + 0.2 / math.sqrt(252) * rng.gauss(0, 1)))
        for step, spot in enumerate(spots):
            batch["marks"].extend([spot, call_price(spot, (84 - step) / 252)])
            if step < 2:
                batch["features"].extend([math.log(spot / 100) / 0.02, (2 - step) / 2])
        batch["payoffs"].append(abs(spots[-1] - 100))
    return batch


def run(url: str, paths: int = 64, updates: int = 30, checkpoint_path: Path | None = None) -> dict:
    client = JormungandrClient(url)
    model_id = "hedge-" + uuid.uuid4().hex[:12]
    premium = (2 * call_price(100, 2 / 252) - 100 * math.exp(-0.01 * 2 / 252)
               + 100 * math.exp(-0.03 * 2 / 252))
    config = {"target": "cpu", "precision": "f64", "seed": 23,
              "learning_rate": 0.001, "objective": {"kind": "squared_loss"},
              "premium": premium, "trading_costs": [0.0005, 0.01],
              "liquidation_costs": [0.0005, 0.01], "cash_growth": [math.exp(0.03 / 252)] * 2,
              "income_growth": [math.exp(0.01 / 252), 1.0] * 2}
    client.create_native_model({"model_id": model_id, "backend": "volt_burn", "config": config})
    try:
        client.native_command(model_id, "register_dataset", {
            "dataset_id": "gbm-study", "partitions": [
                {"split": "train", "first_path": 0, "paths": paths},
                {"split": "validation", "first_path": paths, "paths": paths}]})
        training_batch = scenario_batch(paths, 0, 123)
        validation_batch = scenario_batch(paths, paths, 999)
        for batch_id, split, batch in [("train", "train", training_batch), ("validation", "validation", validation_batch)]:
            client.native_command(model_id, "upload", {"batch_id": batch_id, "split": split, "batch": batch})
        before = client.native_command(model_id, "evaluate", {"batch_id": "validation"})
        for _ in range(updates):
            version = client.get_native_model(model_id)["version"]
            client.native_command(model_id, "update", {"batch_id": "train", "expected_version": version})
        after = client.native_command(model_id, "evaluate", {"batch_id": "validation"})
        targets = client.native_command(model_id, "infer", {
            "step": 0, "paths": 1, "features": [0.0, 1.0], "previous_holdings": [0.0, 0.0]})
        checkpoint = client.native_command(model_id, "checkpoint")
        checkpoint.pop("ok", None)
        if checkpoint_path is not None:
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2) + "\n")
        restored_id = model_id + "-restored"
        client.create_native_model({"model_id": restored_id, "backend": "volt_burn", "checkpoint": checkpoint})
        try:
            restored = client.native_command(restored_id, "infer", {
                "step": 0, "paths": 1, "features": [0.0, 1.0], "previous_holdings": [0.0, 0.0]})
            assert restored["holdings"] == targets["holdings"]
        finally:
            client.delete_native_model(restored_id)
        return {"paths_per_batch": paths, "updates": updates, "initial_validation_loss": before["loss"],
                "final_validation_loss": after["loss"], "version": after["version"],
                "inference": targets, "checkpoint_saved": checkpoint_path is not None,
                "checkpoint_has_optimizer": all(
                    {"first_moment", "second_moment", "adam_step"} <= set(p)
                    for p in checkpoint["policy"]["parameters"]),
                "restore_inference_matches": restored["holdings"] == targets["holdings"]}
    finally:
        client.delete_native_model(model_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--paths", type=int, default=64)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.paths <= 512 or not 1 <= args.updates <= 10000:
        parser.error("paths must be 1..512 and updates must be 1..10000")
    result = run(args.url, args.paths, args.updates, args.checkpoint)
    report = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(report)
    print(report, end="")
