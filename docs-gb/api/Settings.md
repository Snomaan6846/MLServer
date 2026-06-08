# Settings

### Config

| Attribute | Type | Default |
|-----------|------|---------|
| `extra` | `str` | `"ignore"` |
| `env_prefix` | `str` | `"MLSERVER_"` |
| `env_file` | `str` | `".env"` |
| `protected_namespaces` | `tuple` | `()` |

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cache_enabled` | `bool` | `False` | Enable caching for the model predictions. |
| `cache_size` | `int` | `100` | Cache size to be used if caching is enabled. |
| `cors_settings` | `CORSSettings \| None` | `None` | - |
| `debug` | `bool` | `False` | - |
| `empty_registry_readiness` | `bool` | `True` | Flag to control readiness check behavior when no models are loaded. If True (default), the server reports ready when the model registry is empty. If False, the server reports not ready when the model registry is empty. **Note:** This flag only applies AFTER server startup completes. During startup, an empty registry always reports not ready to prevent race conditions. |
| `environments_dir` | `str` | `'-'` | - |
| `extensions` | `list[str]` | `[]` | - |
| `grpc_max_message_length` | `int \| None` | `None` | - |
| `grpc_port` | `int` | `8081` | - |
| `gzip_enabled` | `bool` | `True` | Enable GZipMiddleware. |
| `host` | `str` | `'0.0.0.0'` | - |
| `http_port` | `int` | `8080` | - |
| `kafka_enabled` | `bool` | `False` | Enable Kafka integration for the server. |
| `kafka_servers` | `str` | `'localhost:9092'` | Comma-separated list of Kafka servers. |
| `kafka_topic_input` | `str` | `'mlserver-input'` | Kafka topic for input messages. |
| `kafka_topic_output` | `str` | `'mlserver-output'` | Kafka topic for output messages. |
| `load_models_at_startup` | `bool` | `True` | - |
| `logging_settings` | `str \| dict[Any, Any] \| None` | `None` | Path to logging config file or dictionary configuration. |
| `metrics_dir` | `str` | `'-'` | Directory used to share metrics across parallel workers. Equivalent to the `PROMETHEUS_MULTIPROC_DIR` env var in `prometheus-client`. Note that this won't be used if the `parallel_workers` flag is disabled. By default, the `.metrics` folder of the current working directory will be used. |
| `metrics_endpoint` | `str \| None` | `'/metrics'` | Endpoint used to expose Prometheus metrics. Alternatively, can be set to `None` to disable it. |
| `metrics_port` | `int` | `8082` | Port used to expose metrics endpoint. |
| `metrics_rest_server_prefix` | `str` | `'rest_server'` | Metrics rest server string prefix to be exported. |
| `model_repository_implementation` | `ImportString \| None` | `None` | - |
| `model_repository_implementation_args` | `dict` | `{}` | - |
| `model_repository_root` | `str` | `'.'` | - |
| `parallel_workers` | `int` | `1` | - |
| `parallel_workers_timeout` | `int` | `5` | - |
| `root_path` | `str` | `''` | - |
| `strict_readiness` | `bool` | `True` | Flag to control readiness check behavior for loaded models. If True (default), ALL models must be ready for health check to pass. If False, AT LEAST ONE model must be ready for health check to pass. **Note:** During server startup, readiness is always strict (all models must be ready) regardless of this setting. After startup completes, this setting takes effect. |
| `server_name` | `str` | `'mlserver'` | - |
| `server_version` | `str` | `'1.7.0.dev0'` | - |
| `tracing_server` | `str \| None` | `None` | Server name used to export OpenTelemetry tracing to collector service. |
| `use_structured_logging` | `bool` | `False` | Use JSON-formatted structured logging instead of default format. |
