# GENERATED from schemas/jormungandr_inference_schema.json; DO NOT EDIT.
from __future__ import annotations

import struct
from typing import Any, Mapping

import numpy as np

SCHEMA_FORMAT = "jormungandr-inference-protocol-schema-v1"
SCHEMA_SHA256 = "f899b6473981d7c28e257f9291172998ea216e54ce5b08e3b2a405eeced4a3de"
SCHEMA_ID = 902
SCHEMA_VERSION = 1
HEADER = struct.Struct("<6H")
VAR_DATA_LENGTH = struct.Struct("<I")
REQUEST_TEMPLATE_ID = 1
REQUEST_BLOCK_LENGTH = 20
REQUEST_FIXED = struct.Struct("<QfIB3x")
RESPONSE_TEMPLATE_ID = 2
RESPONSE_BLOCK_LENGTH = 48
RESPONSE_FIXED = struct.Struct("<QIfQQiIII")
DETERMINISTIC_FLAG = 0x01


def _header(template_id: int, block_length: int, var_data_fields: int) -> bytes:
    return HEADER.pack(
        block_length,
        template_id,
        SCHEMA_ID,
        SCHEMA_VERSION,
        0,
        var_data_fields,
    )


def _decode_header(
    data: bytes,
    template_id: int,
    block_length: int,
    var_data_fields: int,
) -> int:
    if len(data) < HEADER.size:
        raise ValueError("truncated SBE message header")
    actual = HEADER.unpack_from(data, 0)
    expected = (
        block_length,
        template_id,
        SCHEMA_ID,
        SCHEMA_VERSION,
        0,
        var_data_fields,
    )
    if actual != expected:
        raise ValueError(
            f"SBE message header mismatch: got {actual}, expected {expected}"
        )
    return HEADER.size


def _float_bytes(values: Any) -> tuple[np.ndarray, bytes]:
    array = np.asarray(values, dtype="<f4").reshape(-1)
    return array, array.tobytes(order="C")


def _var_data(payload: bytes) -> bytes:
    return VAR_DATA_LENGTH.pack(len(payload)) + payload


def _read_float_var_data(
    data: bytes,
    offset: int,
    count: int,
) -> tuple[np.ndarray, int]:
    if len(data) < offset + VAR_DATA_LENGTH.size:
        raise ValueError("truncated SBE varData length")
    byte_length = VAR_DATA_LENGTH.unpack_from(data, offset)[0]
    offset += VAR_DATA_LENGTH.size
    if byte_length != count * 4 or len(data) < offset + byte_length:
        raise ValueError("SBE float varData length/count mismatch")
    values = np.frombuffer(data, dtype="<f4", count=count, offset=offset)
    return values, offset + byte_length


def encode_policy_infer_request(
    obs: Any,
    deterministic: bool,
    epsilon: float,
    request_id: int = 0,
) -> bytes:
    values, payload = _float_bytes(obs)
    fixed = REQUEST_FIXED.pack(
        int(request_id),
        float(epsilon),
        int(values.size),
        DETERMINISTIC_FLAG if deterministic else 0,
    )
    return (
        _header(REQUEST_TEMPLATE_ID, REQUEST_BLOCK_LENGTH, 1)
        + fixed
        + _var_data(payload)
    )


def decode_policy_infer_request(data: bytes) -> Mapping[str, Any]:
    offset = _decode_header(
        data, REQUEST_TEMPLATE_ID, REQUEST_BLOCK_LENGTH, 1
    )
    if len(data) < offset + REQUEST_FIXED.size:
        raise ValueError("truncated SBE PolicyInferRequest fixed block")
    request_id, epsilon, count, flags = REQUEST_FIXED.unpack_from(data, offset)
    offset += REQUEST_BLOCK_LENGTH
    obs, offset = _read_float_var_data(data, offset, int(count))
    if offset != len(data):
        raise ValueError("trailing bytes in SBE PolicyInferRequest")
    return {
        "request_id": int(request_id),
        "epsilon": float(epsilon),
        "deterministic": bool(flags & DETERMINISTIC_FLAG),
        "obs": obs,
    }


def encode_policy_infer_response(
    request_id: int,
    payload: Mapping[str, Any],
) -> bytes:
    q_values, q_payload = _float_bytes(payload.get("q_values", []))
    decision_values, decision_payload = _float_bytes(
        payload.get("decision_values", [])
    )
    aux_probs, aux_payload = _float_bytes(payload.get("aux_probs", []))
    fixed = RESPONSE_FIXED.pack(
        int(request_id),
        int(payload.get("action_idx", 0)),
        float(payload.get("action", 0.0)),
        int(payload.get("policy_version", 0)),
        int(payload.get("updates", 0)),
        int(payload.get("inference_replica_id", -1)),
        int(q_values.size),
        int(decision_values.size),
        int(aux_probs.size),
    )
    return b"".join(
        (
            _header(RESPONSE_TEMPLATE_ID, RESPONSE_BLOCK_LENGTH, 3),
            fixed,
            _var_data(q_payload),
            _var_data(decision_payload),
            _var_data(aux_payload),
        )
    )


def decode_policy_infer_response(data: bytes) -> Mapping[str, Any]:
    offset = _decode_header(
        data, RESPONSE_TEMPLATE_ID, RESPONSE_BLOCK_LENGTH, 3
    )
    if len(data) < offset + RESPONSE_FIXED.size:
        raise ValueError("truncated SBE PolicyInferResponse fixed block")
    values = RESPONSE_FIXED.unpack_from(data, offset)
    offset += RESPONSE_BLOCK_LENGTH
    (
        request_id,
        action_idx,
        action,
        policy_version,
        updates,
        replica_id,
        q_count,
        decision_count,
        aux_count,
    ) = values
    q_values, offset = _read_float_var_data(data, offset, int(q_count))
    decision_values, offset = _read_float_var_data(
        data, offset, int(decision_count)
    )
    aux_probs, offset = _read_float_var_data(data, offset, int(aux_count))
    if offset != len(data):
        raise ValueError("trailing bytes in SBE PolicyInferResponse")
    return {
        "request_id": int(request_id),
        "action_idx": int(action_idx),
        "action": float(action),
        "policy_version": int(policy_version),
        "updates": int(updates),
        "inference_replica_id": int(replica_id),
        "q_values": q_values.tolist(),
        "decision_values": decision_values.tolist(),
        "aux_probs": aux_probs.tolist(),
    }


__all__ = [
    "SCHEMA_FORMAT",
    "SCHEMA_ID",
    "SCHEMA_SHA256",
    "decode_policy_infer_request",
    "decode_policy_infer_response",
    "encode_policy_infer_request",
    "encode_policy_infer_response",
]
