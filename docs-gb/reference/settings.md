# MLServer Settings

MLServer can be configured through a `settings.json` file on the root folder
from where MLServer is started.
Note that these are server-wide settings (e.g. gRPC or HTTP port) which are
separate from the [invidual model settings](./model-settings).
Alternatively, this configuration can also be passed through **environment
variables** prefixed with `MLSERVER_` (e.g. `MLSERVER_GRPC_PORT`).
By default, debug logging is disabled (`"debug": false`) and can be enabled with
`MLSERVER_DEBUG=1` or by setting `"debug": true` in `settings.json`.

## Settings

```{eval-rst}

.. autopydantic_settings:: mlserver.settings.Settings
```
