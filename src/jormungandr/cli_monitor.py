"""Watch a running Jörmungandr structured learner over its public API."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
import time
from typing import Any, Mapping

from jormungandr.client import JormungandrClient, JormungandrClientError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8811")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _view(model: Mapping[str, Any]) -> dict[str, Any]:
    replay = model.get("replay", {})
    experience = model.get("experience", {})
    return {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_id": model.get("model_id"),
        "algorithm": (model.get("algorithm") or {}).get("name"),
        "device": model.get("device"),
        "replay": f"{replay.get('size', 0)}/{replay.get('capacity', 0)}",
        "experience_items": experience.get("items", 0),
        "updates": model.get("updates", 0),
        "policy_version": model.get("policy_version", 0),
        "inference_calls": model.get("inference_calls", 0),
        "last_loss": model.get("last_loss"),
        "last_td_abs": model.get("last_td_abs"),
        "last_metrics": model.get("last_metrics", {}),
        "last_error": model.get("last_error", ""),
    }


def _select_model(client: JormungandrClient, model_id: str) -> str:
    if model_id:
        return model_id
    models = client.list_structured_models().get("models", [])
    if len(models) != 1:
        identifiers = [str(item.get("model_id", "")) for item in models]
        raise JormungandrClientError(
            "--model-id is required unless exactly one model is active; "
            f"found {identifiers}"
        )
    return str(models[0]["model_id"])


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    client = JormungandrClient(args.url, timeout=max(2.0, args.interval * 2.0))
    try:
        model_id = _select_model(client, str(args.model_id).strip())
        while True:
            model = client.get_structured_model(model_id)["model"]
            view = _view(model)
            if args.json or not sys.stdout.isatty():
                print(json.dumps(view, sort_keys=True), flush=True)
            else:
                print("\033[2J\033[H", end="")
                print(f"Jörmungandr learner — {view['time']}")
                print(f"model          {view['model_id']}")
                print(f"algorithm      {view['algorithm']} on {view['device']}")
                print(f"replay         {view['replay']}")
                print(f"experience     {view['experience_items']}")
                print(f"updates        {view['updates']}")
                print(f"policy version {view['policy_version']}")
                print(f"inference      {view['inference_calls']}")
                print(f"loss / |TD|    {view['last_loss']} / {view['last_td_abs']}")
                print(f"metrics        {json.dumps(view['last_metrics'], sort_keys=True)}")
                print(f"last error     {view['last_error'] or '-'}")
                print("\nCtrl-C to stop monitoring (training continues).", flush=True)
            if args.once:
                return 0
            time.sleep(max(0.2, float(args.interval)))
    except KeyboardInterrupt:
        return 0
    except JormungandrClientError as exc:
        print(f"jormungandr-monitor: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
