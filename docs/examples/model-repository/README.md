# Multi-model serving and the Model Repository API

A single MLServer process can serve multiple models (and multiple versions of
the same model) under different paths. This example uses two models:

| Name               | Framework      | Path                              |
| ------------------ | -------------- | --------------------------------- |
| `mnist-svm`        | `scikit-learn` | `./models/mnist-svm/`             |
| `mushroom-xgboost` | `xgboost`      | `./models/mushroom-xgboost/`     |

That layout is multi-model serving: both models are loaded at startup and
reachable on their own inference URLs.

On top of that, MLServer exposes a **Model Repository API** to list, unload,
and load models at runtime. It follows [Triton's Model Repository
extension](https://github.com/triton-inference-server/server/blob/master/docs/protocol/extension_model_repository.md)
to the V2 dataplane.

Training for these models is covered in the [Scikit-Learn](../sklearn/README.md)
and [XGBoost](../xgboost/README.md) examples. This notebook ships the serialised
artifacts and focuses on serving both together, then managing them through the
repository API.

## Models

The repository is a folder per model, each with its own `model-settings.json`:

- `settings.json`: server-wide config (ports, log level, …)
- `models/mnist-svm/model-settings.json`: sklearn runtime for `mnist-svm`
- `models/mushroom-xgboost/model-settings.json`: xgboost runtime for `mushroom-xgboost`


```python
!ls -R ./models
```

## Serving

Start MLServer from this directory. By default it **loads every model** in the
repository.

```shell
mlserver start .
```

## List available models

Both models should now be `READY`.


```python
import requests

response = requests.post("http://localhost:8080/v2/repository/index", json={})
response.json()
```

The index lists `mushroom-xgboost` and `mnist-svm`. `READY` means they are
loaded and available for inference.

## Infer against both models

Each model is served on its own V2 path. The payloads below match the test
samples from the sklearn digits and XGBoost agaricus examples.


```python
import requests

digit = [
    0.0, 0.0, 1.0, 11.0, 14.0, 15.0, 3.0, 0.0,
    0.0, 1.0, 13.0, 16.0, 12.0, 16.0, 8.0, 0.0,
    0.0, 8.0, 16.0, 4.0, 6.0, 16.0, 5.0, 0.0,
    0.0, 5.0, 15.0, 11.0, 13.0, 14.0, 0.0, 0.0,
    0.0, 0.0, 2.0, 12.0, 16.0, 13.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 13.0, 16.0, 16.0, 6.0, 0.0,
    0.0, 0.0, 0.0, 16.0, 16.0, 16.0, 7.0, 0.0,
    0.0, 0.0, 0.0, 11.0, 13.0, 12.0, 1.0, 0.0,
]

mnist_request = {
    "inputs": [
        {
            "name": "predict",
            "shape": [1, 64],
            "datatype": "FP32",
            "data": digit,
        }
    ]
}

response = requests.post(
    "http://localhost:8080/v2/models/mnist-svm/versions/v0.1.0/infer",
    json=mnist_request,
)
response.json()
```


```python
# First agaricus test row as a dense 126-feature vector. SVMLight files use
# 1-based indexes; sklearn.load_svmlight_file converts them to 0-based columns.
mushroom = [0.0] * 126
for idx in [
    0, 8, 18, 20, 23, 33, 35, 38, 41, 52, 55, 64,
    68, 76, 85, 87, 91, 94, 101, 105, 116, 121,
]:
    mushroom[idx] = 1.0

xgboost_request = {
    "inputs": [
        {
            "name": "predict",
            "shape": [1, 126],
            "datatype": "FP32",
            "data": mushroom,
        }
    ]
}

response = requests.post(
    "http://localhost:8080/v2/models/mushroom-xgboost/versions/v0.1.0/infer",
    json=xgboost_request,
)
response.json()
```

## Unloading `mushroom-xgboost`

Unload one model. It stays on disk in the repository, but it is no longer
served.


```python
requests.post("http://localhost:8080/v2/repository/models/mushroom-xgboost/unload")
```

The index should now flag `mushroom-xgboost` as `UNAVAILABLE`.


```python
response = requests.post("http://localhost:8080/v2/repository/index", json={})
response.json()
```

`mnist-svm` remains available. A request to the unloaded model should fail.


```python
response = requests.post(
    "http://localhost:8080/v2/models/mushroom-xgboost/versions/v0.1.0/infer",
    json=xgboost_request,
)
response.status_code, response.text
```

## Loading `mushroom-xgboost` back


```python
requests.post("http://localhost:8080/v2/repository/models/mushroom-xgboost/load")
```

The index should show both models `READY` again.


```python
response = requests.post("http://localhost:8080/v2/repository/index", json={})
response.json()
```
