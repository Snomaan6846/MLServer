# Inference Performance Options

MLServer provides optional paths for reducing request conversion, frontend JSON
processing and inference IPC overhead. Each option can be enabled independently.
Choose settings using your model, request sizes and concurrency; compare
throughput, tail latency, CPU use and memory before adopting a configuration.

## Startup Heap Freeze

At startup — after settings load and before any model is created — MLServer
calls `gc.freeze()`, moving the interpreter's import heap into the garbage
collector's permanent generation. Request-driven collections then skip
rescanning those objects, reducing GC pauses under load (the same pattern
documented by Instagram's `gc.freeze` deployment and subsequent
tail-latency write-ups). The freeze deliberately runs *before* model creation:
frozen objects are excluded from cyclic collection, so a model frozen at
startup could never be reclaimed after unload. The regression test
`tests/test_model_reclamation.py` pins both the freeze and that boundary.

## Server Settings

These settings apply to the whole server and belong in `settings.json` at the
root of the model repository. They can also be set with environment variables.

| Setting | Default | Environment variable |
| --- | --- | --- |
| `grpc_raw_input_buffers` | `false` | `MLSERVER_GRPC_RAW_INPUT_BUFFERS` |
| `rest_worker_json` | `false` | `MLSERVER_REST_WORKER_JSON` |
| `rest_fast_parse` | `false` | `MLSERVER_REST_FAST_PARSE` |
| `rest_json_compact_floats` | `false` | `MLSERVER_REST_JSON_COMPACT_FLOATS` |
| `rest_binary_tensor` | `false` | `MLSERVER_REST_BINARY_TENSOR` |
| `rest_worker_binary` | `false` | `MLSERVER_REST_WORKER_BINARY` |
| `parallel_inference_transport` | `"queue"` | `MLSERVER_PARALLEL_INFERENCE_TRANSPORT` |
| `frontend_workers` | `0` | `MLSERVER_FRONTEND_WORKERS` |
| `frontend_max_restarts` | `5` | `MLSERVER_FRONTEND_MAX_RESTARTS` |
| `parallel_routing_rotation` | `1.0` | `MLSERVER_PARALLEL_ROUTING_ROTATION` |
| `inference_timeout` | `null` (unbounded) | `MLSERVER_INFERENCE_TIMEOUT` |

For example, the following enables all four options with two worker processes:

```json
{
  "parallel_workers": 2,
  "grpc_raw_input_buffers": true,
  "rest_worker_json": true,
  "rest_binary_tensor": true,
  "parallel_inference_transport": "zmq"
}
```

The ZeroMQ transport ships with MLServer (`pyzmq` is a direct dependency), so
no extra install step is needed. Start the server normally:

```bash
poetry run mlserver start .
```

Restart MLServer after changing server settings. Choose the worker count for
your CPU and model-memory budget; see [Parallel Inference](./parallel-inference.md).

### Raw FP32 gRPC Inputs

`grpc_raw_input_buffers=true` preserves unary gRPC input from
`raw_input_contents` as NumPy buffers through dispatch. It avoids constructing a
large list of Python floats in the frontend. The worker's codec still prepares
the final model input, and IPC still serializes and copies data.

Enable this only when the input codecs of models receiving raw FP32 requests
accept NumPy-backed tensor data, such as the ONNX runtime's default NumPy codec.
Custom runtimes that require lists should leave it disabled. Because the flag
is server-wide, check compatibility across all models served by the instance.

All inputs must have complete, correctly sized raw FP32 contents to use this
path. Other representations and streaming use standard conversion. Models
with batching settings recover list-based inputs before the batcher
merges them (in the workers when parallel inference is enabled).

### REST JSON in Workers

`rest_worker_json=true` forwards the original JSON body to the selected model's
worker. Parsing, validation and response JSON encoding run there, distributing
that CPU work across worker processes.

This applies to unary `/infer` routes, including versioned routes, for models
using parallel inference. Both `max_batch_size` and `max_batch_time` must be `0`
in the model settings for worker JSON processing. Batched models, models without
parallel workers and other media types retain frontend preparation. Generate
and streaming routes retain their existing behavior.

Built-in inference success/failure counters and duration summaries remain owned
by the frontend, so worker replacement does not discard completed requests.
Labels use the model name and version supplied in the request URL; an
unversioned request retains `version="None"`. Valid requests rejected during
model lookup or readiness checking count as inference failures. Body/schema
validation errors do not enter inference accounting.

Duration includes model lookup, worker dispatch/wait and inference. As on the
standard path, it excludes JSON parsing/validation and response JSON encoding.
Workers report those conversion durations on success and error replies, and
the frontend excludes them from its elapsed-time observation. HTTP body receipt
and decompression also remain outside this metric.

Success means inference completed, rather than HTTP response delivery succeeded.
An error formatting that completed prediction retains the inference success,
as on the standard path.

### Native Fast JSON Parsing

`rest_fast_parse=true` parses numeric tensor `data` arrays directly from the
raw request bytes into the target buffer with a compiled C parser, skipping
the intermediate Python list of floats. It requires `rest_worker_json` (raw
bytes only reach a parse site on the worker JSON path) and applies to the
same unary `/infer` routes and non-batched parallel models.

The datatypes FP32, FP64, INT8-64, UINT8-64 and BOOL are handled natively;
every other datatype keeps the standard parse. The fast path is a strict
subset of the standard path: any payload it cannot reproduce exactly —
unsupported datatypes, JSON grammar it does not accept, values whose
semantics differ — falls back to the standard parse, so responses and error
bodies are unchanged.

#### Where the native parser comes from

The wheel is pure Python (`py3-none-any`) and ships the parser's C source
(`mlserver/codecs/fastspan/parser.c`) rather than a compiled library, so
the same wheel installs on every platform. The compiled library
(`libfastspan.so`) is built per platform at image-build time:

- **Official images** (and images derived from them): the library is
  already compiled into the installed package tree during the image build.
  Nothing to do.
- **Serving from a checkout**: run `make fastspan`.
- **A plain `pip install` outside the official images**: the library is
  not compiled by installation. If you enable `rest_fast_parse`, compile
  it once where MLServer is installed (requires a C compiler):

  ```sh
  PKG_DIR=$(python3 -c "import mlserver.codecs.fastspan as m, pathlib; print(pathlib.Path(m.__file__).parent)")
  gcc -O3 -shared -fPIC -o "$PKG_DIR/libfastspan.so" "$PKG_DIR/parser.c" -lm
  ```

Enabling the flag without the library is a startup error naming the
missing file and the fix, not a silent fallback to the standard parse.

Decoded inputs are NumPy views of the parsed buffer: repeated `decode()`
calls on the same input return views of one buffer, so a model that decodes
an input, mutates it in place and decodes it again observes its own
mutation. List-based parsing hands out a fresh copy per decode; models
that decode each input once — the common pattern — are unaffected.
Models using adaptive batching are also unaffected, because array-backed
inputs are restored to the list representation before the batcher merges
them.

### Compact Float Rendering

`rest_json_compact_floats=true` renders FP32 tensor data on REST JSON
responses with the shortest float32 round-trip representation instead of
the default widened float64 text. Responses become roughly 50% smaller
and slightly faster to serialize.

This is the only setting that changes response TEXT: the float32 values
are identical when parsed, but the JSON bytes differ from the default
widened form. Enable it only if consumers compare parsed values rather
than response bodies — a client that byte-compares responses (a cache
key over the raw body, a wire-parity check) will see different text.
`rest_json_compact_floats=false` (the default) keeps the wire
byte-identical to the standard rendering.

### REST Binary Tensor Data

`rest_binary_tensor=true` enables the KServe v2 Binary Tensor Data
Extension on unary REST inference routes. Requests send a JSON inference
header followed by raw tensor bytes, and responses follow the client's
per-output binary/JSON negotiation, avoiding the construction of large
Python float lists in the frontend in both directions.

A binary request carries the length of its JSON header in the
`Inference-Header-Content-Length` header, the `InferenceRequest` object
in the first bytes of the body, and raw tensor data in the remainder.
Numeric tensors arrive as flat NumPy views of the body; BYTES tensors use
the KServe length-prefixed element framing. An output is encoded as raw
bytes when the request-wide `binary_data_output` parameter, or that
output's own `binary_data` parameter, selects binary; the raw data is
appended after a JSON footer carrying a `binary_data_size` parameter per
binary output. Outputs that negotiate JSON keep inline data, so a request
can mix both. Malformed binary frames return `400` responses.

Enable this only when the input codecs of models receiving binary requests
accept NumPy-backed tensor data, such as the ONNX runtime's default NumPy
codec — the same compatibility consideration as raw FP32 gRPC inputs.
Because the setting is server-wide, check compatibility across all models
served by the instance. Requests without
`Inference-Header-Content-Length` keep the standard JSON path.

Unlike worker JSON processing, binary requests remain compatible with
models that use adaptive batching: array-backed inputs are restored to
the public list representation before frontend batching, and mixed BYTES
batches are normalized to per-contract elements before merging. When
`rest_worker_json` is also enabled, JSON bodies keep the worker JSON path
unless the request may negotiate binary outputs; binary frames always
take the binary endpoint. The server advertises `binary_tensor_data` in
the `extensions` list of `GET /v2` while this setting is enabled.

### REST Binary Tensor Data in Workers

`rest_worker_binary=true` (requires `rest_binary_tensor=true`) forwards
binary tensor frames to the selected model's worker as raw bytes. The
worker parses the JSON header, runs inference and encodes the negotiated
response, distributing that CPU work across worker processes — the
binary counterpart of worker JSON processing. Both inference IPC
crossings carry raw bytes instead of Python float lists.

This applies to unary `/infer` routes for models using parallel
inference with `max_batch_size` and `max_batch_time` both `0`. Batched
models, models without parallel workers and JSON requests keep frontend
preparation; JSON bodies follow `rest_worker_json` regardless of this
setting. Inference accounting stays owned by the frontend, including the
worker's conversion durations, and a malformed frame returns `400` even
when model lookup fails — matching the JSON path's error priority.

### Ndarray-Native Response Serialisation

Models whose outputs are NumPy arrays (such as the ONNX runtime's default
codec) get ndarray-native JSON serialisation on the response path: tensor
data held as an ndarray serialises directly to the flat JSON list of the
V2 wire format (matching the output's `shape` field) without building
intermediate Python float lists in the frontend. `NaN` entries serialise
as `null`, since JSON has no NaN representation. This needs no setting;
it is automatic whenever the response tensors are array-backed. The gRPC
converter has an equivalent ndarray-native path, where `NaN` serialises
natively in protobuf.

REST error responses, including `422` body-validation errors, are rendered
with the same `orjson`-based encoder as successful responses, so error
paths avoid the slower default JSON encoding while keeping the same
bodies and status codes.

### ZeroMQ Inference Transport

`parallel_inference_transport="zmq"` uses local ZeroMQ IPC sockets for inference
requests and replies when parallel inference is enabled. PyZMQ ships as a direct
dependency. Model load/unload messages continue
to use multiprocessing queues. Pickle serialization and round-robin worker
selection remain in use; this transport does not provide shared-memory or
zero-copy dispatch.

The JSON and gRPC conversion options work with either transport. Compare
`"queue"` and `"zmq"` at representative loads, including low concurrency. To
return to the default transport, set `parallel_inference_transport` to
`"queue"` **and reset `frontend_workers` to `0`** (the direct server
requires the ZeroMQ transport; the queue transport is parent-child and
cannot be shared), then restart.

## ONNX CUDA Settings

The ONNX runtime also provides `cuda_graph` and `cuda_blocking_sync`, both
defaulting to `false`, under each model's `parameters.extra`. These settings
apply regardless of whether requests arrive over REST or gRPC.

See the [ONNX CUDA tuning guide](../runtimes/onnx.md#cuda-graph-replay-and-blocking-waits)
for a complete model configuration, environment variables, fixed-shape FP32
requirements and the scope of CUDA blocking waits within a process. CUDA Graphs
retain additional device buffers for each model session, so account for every
worker's GPU memory when choosing a worker count.
