# Examples

To see MLServer in action you can check out the examples below.
These are end-to-end notebooks, showing how to serve models with MLServer.

## Inference Runtimes

If you are interested in how MLServer interacts with particular model
frameworks, you can check the following examples.
These focus on showcasing the different [inference
runtimes](../runtimes/index.md) that ship with MLServer out of the box.
Note that, for **advanced use cases**, you can also write your own custom
inference runtime (see the [example below on custom
models](./custom/README.md)).

- [Serving Scikit-Learn models](./sklearn/README.md)
- [Serving XGBoost models](./xgboost/README.md)
- [Serving LightGBM models](./lightgbm/README.md)
- [Serving ONNX models](./onnx/README.md)
- [Serving custom models](./custom/README.md)

```{toctree}
:caption: Inference Runtimes
:titlesonly:
:hidden:

./sklearn/README.md
./xgboost/README.md
./lightgbm/README.md
./onnx/README.md
./custom/README.md
```

## MLServer Features

To see some of the advanced features included in MLServer (e.g. multi-model
serving), check out the examples below.

- [Multi-model serving and the Model Repository API](./model-repository/README.md)
- [Content-Type Decoding](./content-type/README.md)
- [Serving custom models requiring JSON inputs or outputs](./custom-json/README.md)

```{toctree}
:caption: MLServer Features
:titlesonly:
:hidden:

./model-repository/README.md
./content-type/README.md
./custom-json/README.md
```
