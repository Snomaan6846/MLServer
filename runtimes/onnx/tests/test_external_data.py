"""Tests for ONNX external data loading and symlink handling."""

import os
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from mlserver.settings import ModelSettings, ModelParameters
from mlserver.types import InferenceRequest, RequestInput
from mlserver_onnx import OnnxModel
from mlserver_onnx.utils import load_external_tensor_data

from .conftest import TEST_MODEL_IR_VERSION, TEST_MODEL_OPSET_VERSION


def _create_model_with_external_data(model_dir: str) -> str:
    """Build an ONNX model with weights stored as external data.

    Creates a simple MatMul model where the weight tensor is saved to a
    separate .data file alongside the .onnx protobuf.
    """
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 2]
    )

    weights = np.array(
        [[1.0, 0.5], [0.5, 1.0], [1.0, 0.5], [0.5, 1.0]], dtype=np.float32
    )
    weight_initializer = numpy_helper.from_array(weights, name="weights")

    matmul_node = helper.make_node(
        "MatMul", inputs=["input", "weights"], outputs=["output"]
    )

    graph = helper.make_graph(
        [matmul_node],
        "external_data_model",
        [input_tensor],
        [output_tensor],
        initializer=[weight_initializer],
    )

    model = helper.make_model(
        graph,
        producer_name="mlserver-onnx-test",
        opset_imports=[helper.make_opsetid("", TEST_MODEL_OPSET_VERSION)],
        ir_version=TEST_MODEL_IR_VERSION,
    )

    model_path = os.path.join(model_dir, "model.onnx")
    onnx.save(
        model,
        model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=0,
    )
    return model_path


def _create_model_with_multiple_external_files(model_dir: str) -> str:
    """Build a model with each weight tensor in a separate external file."""
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 4]
    )

    weights = np.ones((4, 4), dtype=np.float32)
    bias = np.zeros(4, dtype=np.float32)

    weight_init = numpy_helper.from_array(weights, name="weights")
    bias_init = numpy_helper.from_array(bias, name="bias")

    matmul_node = helper.make_node(
        "MatMul", inputs=["input", "weights"], outputs=["matmul_out"]
    )
    add_node = helper.make_node(
        "Add", inputs=["matmul_out", "bias"], outputs=["output"]
    )

    graph = helper.make_graph(
        [matmul_node, add_node],
        "multi_file_model",
        [input_tensor],
        [output_tensor],
        initializer=[weight_init, bias_init],
    )

    model = helper.make_model(
        graph,
        producer_name="mlserver-onnx-test",
        opset_imports=[helper.make_opsetid("", TEST_MODEL_OPSET_VERSION)],
        ir_version=TEST_MODEL_IR_VERSION,
    )

    model_path = os.path.join(model_dir, "model.onnx")
    onnx.save(
        model,
        model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=False,
        size_threshold=0,
    )
    return model_path


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def external_data_model_dir(tmp_path) -> str:
    """Directory containing an ONNX model with external data."""
    model_dir = str(tmp_path / "ext_model")
    os.makedirs(model_dir)
    return model_dir


@pytest.fixture
def external_data_model_uri(external_data_model_dir) -> str:
    """Path to an ONNX model with external data in a single .data file."""
    return _create_model_with_external_data(external_data_model_dir)


@pytest.fixture
def multi_file_model_uri(tmp_path) -> str:
    """Path to an ONNX model with each tensor in a separate file."""
    model_dir = str(tmp_path / "multi_file_model")
    os.makedirs(model_dir)
    return _create_model_with_multiple_external_files(model_dir)


@pytest.fixture
def symlinked_model_uri(external_data_model_uri, tmp_path) -> str:
    """Symlink a directory containing a model with external data."""
    model_dir = os.path.dirname(external_data_model_uri)
    link_path = str(tmp_path / "symlinked_models")
    os.symlink(model_dir, link_path)
    return os.path.join(link_path, os.path.basename(external_data_model_uri))


@pytest.fixture
def symlinked_file_model_uri(external_data_model_uri, tmp_path) -> str:
    """Symlink individual files (model + data) into a new directory."""
    model_dir = os.path.dirname(external_data_model_uri)
    link_dir = str(tmp_path / "symlinked_files")
    os.makedirs(link_dir)
    for entry in os.listdir(model_dir):
        os.symlink(
            os.path.join(model_dir, entry),
            os.path.join(link_dir, entry),
        )
    return os.path.join(link_dir, os.path.basename(external_data_model_uri))


def _make_settings(model_uri: str) -> ModelSettings:
    return ModelSettings(
        name="ext-data-model",
        implementation=OnnxModel,
        parameters=ModelParameters(uri=model_uri, version="v1.0.0"),
    )


# -- Tests: _load_external_data static method --------------------------------


class TestLoadExternalData:
    """Unit tests for load_external_tensor_data."""

    def test_loads_single_data_file(self, external_data_model_uri):
        """External tensors in a single .data file are loaded into raw_data."""
        model = onnx.load(external_data_model_uri, load_external_data=False)
        base_dir = os.path.dirname(external_data_model_uri)

        for tensor in model.graph.initializer:
            if tensor.data_location == TensorProto.EXTERNAL:
                assert len(tensor.raw_data) == 0

        load_external_tensor_data(model, base_dir)

        for tensor in model.graph.initializer:
            assert tensor.data_location == TensorProto.DEFAULT
            assert len(tensor.raw_data) > 0
            assert len(tensor.external_data) == 0

    def test_loads_multiple_data_files(self, multi_file_model_uri):
        """Each tensor loads from its own external file."""
        model = onnx.load(multi_file_model_uri, load_external_data=False)
        base_dir = os.path.dirname(multi_file_model_uri)

        load_external_tensor_data(model, base_dir)

        loaded_names = []
        for tensor in model.graph.initializer:
            assert tensor.data_location == TensorProto.DEFAULT
            assert len(tensor.raw_data) > 0
            loaded_names.append(tensor.name)

        assert "weights" in loaded_names
        assert "bias" in loaded_names

    def test_preserves_tensor_values(self, external_data_model_uri):
        """Raw bytes match the original weight values after loading."""
        model = onnx.load(external_data_model_uri, load_external_data=False)
        base_dir = os.path.dirname(external_data_model_uri)

        load_external_tensor_data(model, base_dir)

        weight_tensor = next(t for t in model.graph.initializer if t.name == "weights")
        loaded_weights = numpy_helper.to_array(weight_tensor)
        expected = np.array(
            [[1.0, 0.5], [0.5, 1.0], [1.0, 0.5], [0.5, 1.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(loaded_weights, expected)

    def test_noop_without_external_tensors(self, tmp_path):
        """Models with inline data are unchanged by _load_external_data."""
        weights = numpy_helper.from_array(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), name="W"
        )
        input_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 2])
        output_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 2])
        node = helper.make_node("MatMul", ["X", "W"], ["Y"])
        graph = helper.make_graph(
            [node],
            "inline_graph",
            [input_info],
            [output_info],
            initializer=[weights],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", TEST_MODEL_OPSET_VERSION)],
            ir_version=TEST_MODEL_IR_VERSION,
        )
        model_path = str(tmp_path / "inline.onnx")
        onnx.save(model, model_path)

        loaded = onnx.load(model_path, load_external_data=False)
        original_raw = {t.name: bytes(t.raw_data) for t in loaded.graph.initializer}
        assert len(original_raw) > 0, "model must have at least one initializer"

        load_external_tensor_data(loaded, str(tmp_path))

        for tensor in loaded.graph.initializer:
            assert tensor.data_location == TensorProto.DEFAULT
            assert bytes(tensor.raw_data) == original_raw[tensor.name]

    def test_missing_data_file_raises(self, external_data_model_uri):
        """FileNotFoundError when the .data file is missing."""
        model_dir = os.path.dirname(external_data_model_uri)
        model_name = os.path.basename(external_data_model_uri)
        for f in os.listdir(model_dir):
            if f != model_name:
                os.remove(os.path.join(model_dir, f))

        model = onnx.load(external_data_model_uri, load_external_data=False)

        with pytest.raises(FileNotFoundError):
            load_external_tensor_data(model, model_dir)

    def test_reads_with_offset_and_length(self, external_data_model_uri):
        """Tensors with offset/length read the correct byte slice."""
        ref_model = onnx.load(external_data_model_uri, load_external_data=True)
        expected = {
            t.name: numpy_helper.to_array(t).copy() for t in ref_model.graph.initializer
        }

        model = onnx.load(external_data_model_uri, load_external_data=False)
        base_dir = os.path.dirname(external_data_model_uri)

        load_external_tensor_data(model, base_dir)

        for tensor in model.graph.initializer:
            assert tensor.data_location == TensorProto.DEFAULT
            assert len(tensor.raw_data) > 0
            loaded_arr = numpy_helper.to_array(tensor)
            np.testing.assert_array_equal(
                loaded_arr,
                expected[tensor.name],
                err_msg=f"Tensor {tensor.name!r} bytes differ after sliced read",
            )

    def test_offset_exceeds_file_size_raises(self, external_data_model_uri):
        """ValueError when offset exceeds the data file size."""
        model = onnx.load(external_data_model_uri, load_external_data=False)
        base_dir = os.path.dirname(external_data_model_uri)

        for tensor in model.graph.initializer:
            if tensor.data_location == TensorProto.EXTERNAL:
                for entry in tensor.external_data:
                    if entry.key == "offset":
                        entry.value = "999999999"
                        break
                else:
                    tensor.external_data.add(key="offset", value="999999999")
                break

        with pytest.raises(ValueError, match="offset.*exceeds file size"):
            load_external_tensor_data(model, base_dir)

    def test_length_exceeds_available_raises(self, external_data_model_uri):
        """ValueError when length exceeds available data from offset."""
        model = onnx.load(external_data_model_uri, load_external_data=False)
        base_dir = os.path.dirname(external_data_model_uri)

        for tensor in model.graph.initializer:
            if tensor.data_location == TensorProto.EXTERNAL:
                for entry in tensor.external_data:
                    if entry.key == "length":
                        entry.value = "999999999"
                        break
                else:
                    tensor.external_data.add(key="length", value="999999999")
                break

        with pytest.raises(ValueError, match="length.*exceeds available"):
            load_external_tensor_data(model, base_dir)

    def test_loaded_model_is_serializable(self, external_data_model_uri):
        """Model can be serialized to bytes after external data is loaded."""
        model = onnx.load(external_data_model_uri, load_external_data=False)
        base_dir = os.path.dirname(external_data_model_uri)

        load_external_tensor_data(model, base_dir)

        model_bytes = model.SerializeToString()
        assert len(model_bytes) > 0

    def test_loads_attribute_tensor_external_data(self, tmp_path):
        """External data in node attribute tensors (Constant op) is loaded."""
        model_dir = str(tmp_path / "attr_tensor_model")
        os.makedirs(model_dir)

        input_tensor = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [None, 4]
        )
        output_tensor = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [None, 4]
        )

        const_value = helper.make_tensor(
            "const_val",
            TensorProto.FLOAT,
            [1, 4],
            np.ones(4, dtype=np.float32).tolist(),
        )
        constant_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["constant"],
            value=const_value,
        )
        add_node = helper.make_node(
            "Add",
            inputs=["input", "constant"],
            outputs=["output"],
        )

        graph = helper.make_graph(
            [constant_node, add_node],
            "attr_model",
            [input_tensor],
            [output_tensor],
        )
        model = helper.make_model(
            graph,
            producer_name="mlserver-onnx-test",
            opset_imports=[helper.make_opsetid("", TEST_MODEL_OPSET_VERSION)],
            ir_version=TEST_MODEL_IR_VERSION,
        )

        model_path = os.path.join(model_dir, "model.onnx")
        onnx.save(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx.data",
            size_threshold=0,
        )

        loaded = onnx.load(model_path, load_external_data=False)
        load_external_tensor_data(loaded, model_dir)

        model_bytes = loaded.SerializeToString()
        assert len(model_bytes) > 0

        session = ort.InferenceSession(model_bytes)
        result = session.run(
            None,
            {"input": np.array([[1, 2, 3, 4]], dtype=np.float32)},
        )
        np.testing.assert_array_almost_equal(
            result[0], np.array([[2, 3, 4, 5]], dtype=np.float32)
        )

    def test_path_traversal_rejected(self, external_data_model_uri):
        """Absolute paths and '..' components in location are rejected."""
        model = onnx.load(external_data_model_uri, load_external_data=False)
        base_dir = os.path.dirname(external_data_model_uri)

        for tensor in model.graph.initializer:
            if tensor.data_location == TensorProto.EXTERNAL:
                for entry in tensor.external_data:
                    if entry.key == "location":
                        entry.value = "../../etc/shadow"
                        break
                break

        with pytest.raises(ValueError, match="Unsafe external data location"):
            load_external_tensor_data(model, base_dir)

    def test_absolute_path_rejected(self, external_data_model_uri):
        """Absolute paths in location are rejected."""
        model = onnx.load(external_data_model_uri, load_external_data=False)
        base_dir = os.path.dirname(external_data_model_uri)

        for tensor in model.graph.initializer:
            if tensor.data_location == TensorProto.EXTERNAL:
                for entry in tensor.external_data:
                    if entry.key == "location":
                        entry.value = "/etc/passwd"
                        break
                break

        with pytest.raises(ValueError, match="Unsafe external data location"):
            load_external_tensor_data(model, base_dir)

    def test_missing_location_raises(self, external_data_model_uri):
        """EXTERNAL tensor with no location key raises ValueError."""
        model = onnx.load(external_data_model_uri, load_external_data=False)
        base_dir = os.path.dirname(external_data_model_uri)

        for tensor in model.graph.initializer:
            if tensor.data_location == TensorProto.EXTERNAL:
                entries_to_remove = [
                    e for e in tensor.external_data if e.key == "location"
                ]
                for e in entries_to_remove:
                    tensor.external_data.remove(e)
                break

        with pytest.raises(ValueError, match="no 'location' key"):
            load_external_tensor_data(model, base_dir)


# -- Tests: symlink handling --------------------------------------------------


class TestSymlinkLoading:
    """Test loading models through symlinked paths."""

    async def test_load_through_symlinked_directory(self, symlinked_model_uri):
        """Model loads when model dir is a symlink."""
        settings = _make_settings(symlinked_model_uri)
        model = OnnxModel(settings)
        model.ready = await model.load()

        assert model.ready
        assert model._model is not None
        assert model._input_names == ["input"]
        assert model._output_names == ["output"]

    async def test_load_through_symlinked_files(self, symlinked_file_model_uri):
        """Model loads when individual files are symlinks."""
        settings = _make_settings(symlinked_file_model_uri)
        model = OnnxModel(settings)
        model.ready = await model.load()

        assert model.ready
        assert model._model is not None

    async def test_predict_through_symlink(self, symlinked_model_uri):
        """Inference works on a model loaded through symlinks."""
        settings = _make_settings(symlinked_model_uri)
        model = OnnxModel(settings)
        model.ready = await model.load()

        request = InferenceRequest(
            inputs=[
                RequestInput(
                    name="input",
                    shape=[1, 4],
                    data=[1.0, 2.0, 3.0, 4.0],
                    datatype="FP32",
                )
            ]
        )

        response = await model.predict(request)
        assert len(response.outputs) == 1
        assert response.outputs[0].data is not None


# -- Tests: full model load with external data --------------------------------


class TestExternalDataModelLoad:
    """End-to-end tests: OnnxModel.load() with external data models."""

    async def test_load_model_with_external_data(self, external_data_model_uri):
        """OnnxModel.load() succeeds for models with external data."""
        settings = _make_settings(external_data_model_uri)
        model = OnnxModel(settings)
        model.ready = await model.load()

        assert model.ready
        assert model._model is not None
        assert model._input_names == ["input"]
        assert model._output_names == ["output"]

    async def test_load_model_with_multiple_external_files(self, multi_file_model_uri):
        """OnnxModel.load() succeeds with per-tensor external files."""
        settings = _make_settings(multi_file_model_uri)
        model = OnnxModel(settings)
        model.ready = await model.load()

        assert model.ready
        assert model._model is not None

    async def test_predict_with_external_data_model(self, external_data_model_uri):
        """Inference produces correct results for external-data models."""
        settings = _make_settings(external_data_model_uri)
        model = OnnxModel(settings)
        model.ready = await model.load()

        request = InferenceRequest(
            inputs=[
                RequestInput(
                    name="input",
                    shape=[1, 4],
                    data=[1.0, 2.0, 3.0, 4.0],
                    datatype="FP32",
                )
            ]
        )

        response = await model.predict(request)
        assert len(response.outputs) == 1

        output_data = response.outputs[0].data
        expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) @ np.array(
            [[1.0, 0.5], [0.5, 1.0], [1.0, 0.5], [0.5, 1.0]], dtype=np.float32
        )
        np.testing.assert_array_almost_equal(output_data, expected.flatten())

    async def test_unload_after_external_data_load(self, external_data_model_uri):
        """Unload works correctly after loading external data model."""
        settings = _make_settings(external_data_model_uri)
        model = OnnxModel(settings)
        model.ready = await model.load()

        assert model.ready
        assert await model.unload()
        assert model._model is None

    async def test_metadata_extracted_for_external_data_model(
        self, external_data_model_uri
    ):
        """Input/output metadata is correct for external-data models."""
        settings = _make_settings(external_data_model_uri)
        model = OnnxModel(settings)
        model.ready = await model.load()

        assert len(model.inputs) == 1
        assert model.inputs[0].name == "input"
        assert len(model.outputs) == 1
        assert model.outputs[0].name == "output"
