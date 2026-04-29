# ModelParameters

### Config

| Attribute | Type | Default |
|-----------|------|---------|
| `extra` | `str` | `"allow"` |
| `env_prefix` | `str` | `"MLSERVER_MODEL_"` |
| `env_file` | `str` | `".env"` |
| `protected_namespaces` | `tuple` | `('model_', 'settings_')` |

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `autogenerate_inference_pool_gid` | `bool` | `False` | Flag to autogenerate the inference pool group id for this model. |
| `content_type` | `str \| None` | `None` | Default content type to use for requests and responses. |
| `environment_path` | `str \| None` | `None` | Path to a directory that contains the python environment to be used to load this model. |
| `environment_tarball` | `str \| None` | `None` | Path to the environment tarball which should be used to load this model. |
| `extra` | `dict \| None` | `<factory>` | Arbitrary settings, dependent on the inference runtime implementation. |
| `format` | `str \| None` | `None` | Format of the model (only available on certain runtimes). |
| `inference_pool_gid` | `str \| None` | `None` | Inference pool group id to be used to serve this model. |
| `uri` | `str \| None` | `None` | URI where the model artifacts can be found. This path must be either absolute or relative to where MLServer is running. |
| `version` | `str \| None` | `None` | Version of the model. |
