# SBE policy inference

Jörmungandr provides a binary HTTP endpoint for policy inference on fixed-size
observation vectors. The message format uses the FIX Simple Binary Encoding
(SBE) header and little-endian primitive values.

The endpoint is:

```http
POST /v1/models/{model_id}/policy/infer-sbe
Content-Type: application/vnd.fixprotocol.sbe
```

Each request carries one observation. Concurrent requests still share the
service's inference queue and may run together as one model batch.

Schema version 1 selects from the model's full action vocabulary. The JSON
`/v1/models/{model_id}/policy/infer` endpoint accepts action masks and explicit
observation batches. A model with a windowed-X/state layout accepts SBE after
the actor flattens `x` followed by `state`, using the dimensions recorded with
the model.

## Python client

The package includes the codec and HTTP client in its base installation:

```python
from jormungandr.client import JormungandrClient

client = JormungandrClient("http://127.0.0.1:8811")
result = client.infer_policy_sbe(
    "policy-a",
    [0.25, -0.50, 0.75, 1.00],
    deterministic=True,
    epsilon=0.0,
    request_id=42,
)

print(result["action_idx"], result["action"])
print(result["policy_version"], result["inference_replica_id"])
```

The decoded result is a dictionary:

```python
{
    "request_id": 42,
    "action_idx": 1,
    "action": 0.0,
    "policy_version": 17,
    "updates": 16,
    "inference_replica_id": 0,
    "q_values": [...],
    "decision_values": [...],
    "aux_probs": [...],
}
```

`request_id` is copied from the request. It lets a caller associate responses
with its own work. The service returns the value unchanged.

## Raw HTTP use

Applications can use the checked-in encoder and decoder directly:

```python
from urllib.request import Request, urlopen

from jormungandr.generated_sbe import (
    SCHEMA_FORMAT,
    decode_policy_infer_response,
    encode_policy_infer_request,
)

body = encode_policy_infer_request(
    [0.25, -0.50, 0.75, 1.00],
    deterministic=True,
    epsilon=0.0,
    request_id=42,
)
request = Request(
    "http://127.0.0.1:8811/v1/models/policy-a/policy/infer-sbe",
    data=body,
    headers={"Content-Type": "application/vnd.fixprotocol.sbe"},
    method="POST",
)

with urlopen(request, timeout=5.0) as response:
    assert response.headers["X-Jormungandr-Schema"] == SCHEMA_FORMAT
    result = decode_policy_infer_response(response.read())
```

A successful response has status 200 and these headers:

| Header | Value |
| --- | --- |
| `Content-Type` | `application/vnd.fixprotocol.sbe` |
| `Content-Length` | Encoded response length in bytes |
| `X-Jormungandr-Schema` | `jormungandr-inference-protocol-schema-v1` |

Errors use the service's JSON error body:

```json
{"ok":false,"error":"description"}
```

| Status | Meaning |
| --- | --- |
| 400 | Malformed SBE header or body, or an observation width mismatch |
| 404 | Unknown model identifier |
| 413 | Request exceeds the configured compressed or expanded size limit |
| 500 | Inference failed inside the service |

## Common message header

Every message starts with the 12-byte SBE header below. Offsets are measured
from the beginning of the message. All multi-byte fields are little-endian.

| Offset | Type | Field | Version 1 value |
| ---: | --- | --- | ---: |
| 0 | `uint16` | `blockLength` | 20 for a request; 48 for a response |
| 2 | `uint16` | `templateId` | 1 for a request; 2 for a response |
| 4 | `uint16` | `schemaId` | 902 |
| 6 | `uint16` | `version` | 1 |
| 8 | `uint16` | `numGroups` | 0 |
| 10 | `uint16` | `numVarDataFields` | 1 for a request; 3 for a response |

The decoder checks all six header values.

## PolicyInferRequest

Template 1 has a 20-byte fixed block immediately after the common header.
Offsets in this table are relative to the start of that fixed block.

| Offset | Type | Field | Meaning |
| ---: | --- | --- | --- |
| 0 | `uint64` | `requestId` | Caller-supplied correlation value |
| 8 | `float32` | `epsilon` | Exploration rate; built-in value policies clamp it to 0 through 1 |
| 12 | `uint32` | `observationCount` | Number of float32 values in the observation |
| 16 | `uint8` | `flags` | Bit 0 requests deterministic action selection |
| 17 | 3 bytes | padding | Zero-filled by the supplied encoder |

The fixed block is followed by one variable-data field:

```text
uint32 byteLength
float32 observation[observationCount]
```

`byteLength` must equal `observationCount * 4`. The observation values use
row-major order. A request occupies `36 + 4 * observationCount` bytes.

For the built-in value policies, the deterministic flag sets the effective
epsilon to zero. With deterministic selection disabled, epsilon controls the
chance of choosing a uniformly sampled action from the model's action
vocabulary.

## PolicyInferResponse

Template 2 has a 48-byte fixed block immediately after the common header.
Offsets are relative to the start of that block.

| Offset | Type | Field | Meaning |
| ---: | --- | --- | --- |
| 0 | `uint64` | `requestId` | Value copied from the request |
| 8 | `uint32` | `actionIndex` | Selected index in the model's action vocabulary |
| 12 | `float32` | `action` | Action value stored at `actionIndex` |
| 16 | `uint64` | `policyVersion` | Version used for this inference |
| 24 | `uint64` | `updates` | Learner update count reported by that version |
| 32 | `int32` | `replicaId` | Serving replica identifier; `-1` when unavailable |
| 36 | `uint32` | `qValueCount` | Number of values in `qValues` |
| 40 | `uint32` | `decisionValueCount` | Number of values in `decisionValues` |
| 44 | `uint32` | `auxProbabilityCount` | Number of values in `auxProbabilities` |

Three variable-data fields follow in this order:

```text
uint32 qValuesByteLength
float32 qValues[qValueCount]

uint32 decisionValuesByteLength
float32 decisionValues[decisionValueCount]

uint32 auxProbabilitiesByteLength
float32 auxProbabilities[auxProbabilityCount]
```

Each byte length must equal its element count multiplied by four. The response
occupies `72 + 4 * (qValueCount + decisionValueCount +
auxProbabilityCount)` bytes.

The arrays depend on the model plugin:

- `qValues` contains per-action Q summaries when the plugin supplies them.
- `decisionValues` contains the per-action scores used for action selection.
- `auxProbabilities` contains auxiliary-head class probabilities when the model
  has an auxiliary head.

An unavailable array has a count and byte length of zero.

## Schema files and compatibility

The canonical source is
[`schemas/jormungandr_inference_schema.json`](../schemas/jormungandr_inference_schema.json).
The repository also contains:

- [`schemas/jormungandr_inference_sbe.xml`](../schemas/jormungandr_inference_sbe.xml)
  for SBE tooling;
- [`src/jormungandr/generated_sbe.py`](../src/jormungandr/generated_sbe.py) for
  the built-in Python codec.

The canonical schema has format name
`jormungandr-inference-protocol-schema-v1`, schema ID 902, version 1, and
semantic version 1.0.0. Its canonical JSON SHA-256 is
`f899b6473981d7c28e257f9291172998ea216e54ce5b08e3b2a405eeced4a3de`.

The supplied decoder accepts the exact version 1 header, fixed-block lengths,
variable-field counts, array lengths, and total message length. A protocol
change therefore requires an updated schema version and regenerated codecs.

The schema digest, codec round trips, HTTP endpoint, auxiliary probabilities,
and replica metadata are covered by [`tests/test_sbe.py`](../tests/test_sbe.py).
