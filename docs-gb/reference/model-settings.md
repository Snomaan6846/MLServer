# Model Settings

In MLServer, each loaded model can be configured separately.
This configuration will include model information (e.g. metadata about the
accepted inputs), but also model-specific settings (e.g. number of [parallel
workers](../user-guide/parallel-inference) to run inference).

This configuration will usually be provided through a `model-settings.json`
file which **sits next to the model artifacts**.
However, it's also possible to provide this through environment variables
prefixed with `MLSERVER_MODEL_` (e.g. `MLSERVER_MODEL_IMPLEMENTATION`). Note
that, in the latter case, this environment variables will be shared across all
loaded models (unless they get overriden by a `model-settings.json` file).
Additionally, if no `model-settings.json` file is found, MLServer will also try
to load a _"default"_ model from these environment variables.

## Runtime Implementation Security

MLServer validates `implementation` against a trusted allowlist of runtime
classes before importing it.

- Built-in runtimes (for example, `mlserver_sklearn.SKLearnModel`,
  `mlserver_xgboost.XGBoostModel`, `mlserver_lightgbm.LightGBMModel`, and
  `mlserver_onnx.OnnxModel`) are always allowlisted.
- Custom runtimes require both a trusted import path and baked source path at
  image build time. For each custom runtime, pass
  `--allow-runtime module.ClassName` and a matching `--runtime-path`.
- If `--runtime-path` points to a directory, it must be an importable Python
  package containing `__init__.py`.
- The dotted `module.ClassName` format is required intentionally to keep
  runtime declarations explicit and unambiguous.
- The same validation applies regardless of whether `implementation` comes from
  `model-settings.json` or `MLSERVER_MODEL_IMPLEMENTATION`.

### Troubleshooting trusted runtime validation

If startup or model loading fails with:

`Model implementation 'module.ClassName' is not in the allowlist of trusted runtimes.`

check the following:

- The value is a dotted import path in `module.ClassName` format.
- The runtime package and module are importable in the serving image.
- For custom runtimes in built images, include each runtime with
  `mlserver build --allow-runtime module.ClassName` and a matching
  `--runtime-path`.
- If both environment variables and `model-settings.json` are present, remember
  `model-settings.json` values take precedence per model.

### Migration note for existing custom runtimes

If you previously relied on dynamically importable runtime paths without an
explicit allowlist entry, update your image build pipeline to pass all served
custom runtimes through `--allow-runtime` and include corresponding
`--runtime-path` values. This keeps runtime loading explicit and prevents
accidental execution of unexpected classes.

## Settings

```{eval-rst}

.. autopydantic_settings:: mlserver.settings.ModelSettings
```

## Extra Model Parameters

```{eval-rst}

.. autopydantic_settings:: mlserver.settings.ModelParameters
```
