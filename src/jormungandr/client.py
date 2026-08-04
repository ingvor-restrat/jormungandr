"""Small standard-library client for Jörmungandr's public HTTP API."""

from __future__ import annotations

import gzip
import json
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


class JormungandrClientError(RuntimeError):
    pass


class JormungandrClient:
    """Picklable HTTP client suitable for external actor processes."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        compress_threshold_bytes: int = 1_000_000,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        if not self.base_url:
            raise ValueError("base_url is required")
        self.timeout = max(0.1, float(timeout))
        self.compress_threshold_bytes = max(
            0, int(compress_threshold_bytes)
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = (
            None
            if payload is None
            else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        headers = {"Content-Type": "application/json"}
        if (
            data is not None
            and self.compress_threshold_bytes > 0
            and len(data) >= self.compress_threshold_bytes
        ):
            data = gzip.compress(data)
            headers["Content-Encoding"] = "gzip"
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8"))
            except Exception:
                detail = {"error": str(exc)}
            raise JormungandrClientError(
                str(detail.get("detail") or detail.get("error") or exc)
            ) from exc
        except Exception as exc:
            raise JormungandrClientError(str(exc)) from exc
        if not isinstance(result, dict) or not bool(result.get("ok", False)):
            raise JormungandrClientError(
                str(result.get("error", "invalid service response"))
                if isinstance(result, dict)
                else "invalid service response"
            )
        return result

    @staticmethod
    def _model_path(model_id: str) -> str:
        return quote(str(model_id), safe="")

    def create_structured_model(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._request("POST", "/v2/models", payload)

    def list_structured_models(self) -> dict[str, Any]:
        return self._request("GET", "/v2/models")

    def get_structured_model(self, model_id: str) -> dict[str, Any]:
        return self._request(
            "GET", f"/v2/models/{self._model_path(model_id)}"
        )

    def get_structured_metrics(self, model_id: str) -> dict[str, Any]:
        return self._request(
            "GET", f"/v2/models/{self._model_path(model_id)}/metrics"
        )

    def log_structured_metrics(
        self,
        model_id: str,
        *,
        step: int,
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v2/models/{self._model_path(model_id)}/metrics",
            {"step": int(step), "metrics": dict(metrics)},
        )

    def infer_structured(
        self,
        model_id: str,
        observations: Sequence[Mapping[str, Any]],
        *,
        deterministic: bool,
        epsilon: float = 0.0,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v2/models/{self._model_path(model_id)}/policy/infer",
            {
                "observations": list(observations),
                "deterministic": bool(deterministic),
                "epsilon": float(epsilon),
            },
        )

    def score_structured(
        self,
        model_id: str,
        observations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v2/models/{self._model_path(model_id)}/policy/score",
            {"observations": list(observations)},
        )

    def add_structured_experience(
        self,
        model_id: str,
        items: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v2/models/{self._model_path(model_id)}/experience/add",
            {
                "schema": "jormungandr.structured_experience.v1",
                "items": list(items),
            },
        )

    def add_structured_trajectories(
        self,
        model_id: str,
        trajectories: Sequence[Sequence[Mapping[str, Any]]],
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v2/models/{self._model_path(model_id)}/trajectories/add",
            {
                "schema": "jormungandr.structured_trajectories.v1",
                "trajectories": [list(item) for item in trajectories],
            },
        )

    def add_structured_trajectory_sequences(
        self,
        model_id: str,
        sequences: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Add compact trajectories whose observation chains are de-duplicated."""

        return self._request(
            "POST",
            f"/v2/models/{self._model_path(model_id)}/trajectories/add",
            {
                "schema": (
                    "jormungandr.structured_joint_trajectory_sequences.v1"
                ),
                "sequences": [dict(item) for item in sequences],
            },
        )

    def add_structured_supervision(
        self,
        model_id: str,
        items: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v2/models/{self._model_path(model_id)}/supervision/add",
            {
                "schema": "jormungandr.structured_supervision_batch.v1",
                "items": list(items),
            },
        )

    def checkpoint_structured_model(self, model_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v2/models/{self._model_path(model_id)}/policy/checkpoint",
            {},
        )
