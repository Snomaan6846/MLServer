# Frequently Asked Questions

Common questions about MLServer development, configuration, and troubleshooting.

---

## General

### What inference protocol does MLServer implement?

MLServer implements the [V2 Inference Protocol](https://kserve.github.io/website/latest/modelserving/data_plane/v2_protocol/) (also known as the Open Inference Protocol), originally developed by NVIDIA and KServe.
It is exposed over both REST (HTTP/1.1 + JSON) and gRPC (HTTP/2 + Protobuf) transports.

### What is the relationship between MLServer and KServe?

MLServer is a **serving runtime** that can run standalone or as a KServe `ServingRuntime`.
KServe provides the Kubernetes orchestration layer (autoscaling, canary rollouts, InferenceService CRD), while MLServer handles the actual model loading and inference execution.

### Which ML frameworks are supported?

MLServer has runtime plugins for several ML frameworks. Four runtimes are
**shipped in the ODH midstream production images**:

| Runtime | Framework | Import Path |
|---------|-----------|-------------|
| sklearn | scikit-learn | `mlserver_sklearn.SKLearnModel` |
| xgboost | XGBoost | `mlserver_xgboost.XGBoostModel` |
| lightgbm | LightGBM | `mlserver_lightgbm.LightGBMModel` |
| onnx | ONNX Runtime | `mlserver_onnx.OnnxModel` |

Additional **community runtimes** have source code and tests in this
repository but are **not included in production images**. They can be
installed separately via `pip` or baked into custom images with
`mlserver build`:

| Runtime | Framework | Import Path |
|---------|-----------|-------------|
| catboost | CatBoost | `mlserver_catboost.CatboostModel` |
| mlflow | MLflow | `mlserver_mlflow.MLflowRuntime` |
| huggingface | HuggingFace Transformers | `mlserver_huggingface.HuggingFaceRuntime` |
| alibi-detect | Alibi Detect | `mlserver_alibi_detect.AlibiDetectRuntime` |
| alibi-explain | Alibi Explain | `mlserver_alibi_explain.AlibiExplainRuntime` |
| mllib | Apache Spark MLlib | `mlserver_mllib.MLlibModel` |

You can also write custom runtimes by subclassing `mlserver.MLModel`.

---

## Configuration

### How do I configure MLServer?

MLServer reads configuration from three sources (in priority order):

1. **Environment variables** — `MLSERVER_` prefix for server settings, `MLSERVER_MODEL_` for model settings
2. **`settings.json`** — Server-wide configuration file in the model repository root
3. **`model-settings.json`** — Per-model configuration file in each model directory

### How do I serve multiple models?

Place each model in its own subdirectory under the model repository root, each with a `model-settings.json`:

```
models/
├── iris-sklearn/
│   ├── model-settings.json
│   └── model.joblib
├── mushroom-xgboost/
│   ├── model-settings.json
│   └── model.bst
```

MLServer discovers and loads all models at startup when `load_models_at_startup: true` (default).

### How do I serve multiple versions of the same model?

Set the `parameters.version` field in each model's `model-settings.json`.
When a version is not specified in inference requests, MLServer routes to the latest version (determined by lexicographic or numeric comparison).

### Can I load and unload models at runtime?

Yes. Use the Model Repository API:

```bash
# Load a model
curl -X POST http://localhost:8080/v2/repository/models/my-model/load

# Unload a model
curl -X POST http://localhost:8080/v2/repository/models/my-model/unload
```

Reloading an already-loaded model performs a rolling deployment: the new version is loaded before the old one is unloaded, ensuring at least one version is always available.

---

## Performance

### How does parallel inference work?

When `parallel_workers > 1`, MLServer spawns separate Python processes for inference.
Each worker loads its own copy of the model, bypassing the GIL for CPU-bound workloads.
The main process dispatches requests to workers via multiprocessing queues.

### What is adaptive batching?

Adaptive batching groups multiple incoming requests into a single batch for more efficient inference.
It is controlled by two settings in `model-settings.json`:

- **`max_batch_size`** — Maximum requests per batch
- **`max_batch_time`** — Maximum seconds to wait for a full batch

A batch is dispatched when either threshold is reached (whichever comes first).

### How does response caching work?

When `cache_enabled: true`, MLServer caches inference responses keyed on the serialized request payload.
Cache is an in-memory LRU with configurable size (`cache_size`, default: 100).
Individual models can opt out by setting `cache_enabled: false` in their `model-settings.json`.

---

## Troubleshooting

### The server reports "not ready" — what's happening?

Check these in order:

1. **Startup not complete** — During initial model loading, the server always reports not ready. Wait for all models to finish loading.
2. **Model load failure** — Check logs for "Couldn't load model" errors. Failed models are removed from the registry.
3. **`strict_readiness: true`** — All models must be ready. If any model is in a failed state, the server reports not ready.
4. **`empty_registry_readiness: false`** — If no models are loaded and this setting is false, the server reports not ready.

### I get "Model not found" errors

- Verify the model name matches exactly (case-sensitive) what's in `model-settings.json`
- Check that the model directory is under `model_repository_root`
- If using versioned endpoints, verify the version string matches `parameters.version`
- Check logs for model load failures during startup

### gRPC requests fail with RESOURCE_EXHAUSTED

The default gRPC message size limit may be too small for your payloads.
Set `grpc_max_message_length` in `settings.json`:

```json
{
  "grpc_max_message_length": 104857600
}
```

This sets the limit to 100 MB (value is in bytes).

### How do I debug inference errors?

1. Enable debug mode: `MLSERVER_DEBUG=true`
2. Check the inference error response body — it contains the exception message
3. Review logs for the full stack trace
4. Use the model-scoped Swagger UI at `/v2/models/{model_name}/docs` to test requests interactively

### Models load slowly in parallel mode

Each parallel worker loads its own copy of the model sequentially.
Loading time scales linearly with `parallel_workers`.
For large models, consider:

- Reducing `parallel_workers` and using adaptive batching instead
- Using memory-mapped model formats (e.g. ONNX, mmap'd pickle)
- Pre-warming the model cache before routing traffic

### How do I handle custom data types?

Implement a custom codec by subclassing `InputCodec` or `RequestCodec`:

```python
from mlserver.codecs import InputCodec, register_input_codec

@register_input_codec
class MyCodec(InputCodec):
    ContentType = "my-type"
    
    @classmethod
    def can_encode(cls, payload) -> bool:
        return isinstance(payload, MyType)
    
    @classmethod
    def encode_output(cls, name, payload, **kwargs):
        # Convert MyType → ResponseOutput
        ...
    
    @classmethod
    def decode_input(cls, request_input, **kwargs):
        # Convert RequestInput → MyType
        ...
```

Then set `content_type: "my-type"` in your request parameters.

---

## Development

### How do I add a new runtime plugin?

See [CONTRIBUTING.md](../../CONTRIBUTING.md#adding-a-new-runtime) for the full guide.
In summary:

1. Create `runtimes/<name>/mlserver_<name>/` with a `runtime.py` subclassing `MLModel`
2. Implement `load()` and `predict()` at minimum
3. Add `pyproject.toml` with the `mlserver` dependency
4. Add an `__init__.py` that exports your runtime class
5. Register the import path in `mlserver/settings.py`

### How do I run tests?

```bash
# Core unit tests
poetry run pytest mlserver/tests/

# Runtime-specific tests
poetry run pytest runtimes/sklearn/tests/

# All tests with coverage
poetry run pytest --cov=mlserver
```

### How do I regenerate the gRPC stubs?

The protobuf stubs are generated from `proto/dataplane.proto`:

```bash
python -m grpc_tools.protoc \
  -I proto \
  --python_out=mlserver/grpc \
  --grpc_python_out=mlserver/grpc \
  proto/dataplane.proto
```

Do not manually edit `dataplane_pb2.py` or `dataplane_pb2_grpc.py` — they are auto-generated.
