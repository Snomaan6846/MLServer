"""Tests for ONNX model lifecycle (load, unload, metadata)."""

import os
import pytest
import onnx

from mlserver_onnx import OnnxModel
from mlserver_onnx.onnx import WELLKNOWN_MODEL_FILENAMES
from mlserver.settings import ModelSettings, ModelParameters

from .conftest import _create_simple_onnx_model


def test_load(model: OnnxModel):
    """Test that model loads successfully."""
    assert model.ready
    assert model._model is not None
    assert model._input_names == ["input"]
    assert model._output_names == ["output"]


async def test_unload(model: OnnxModel):
    """Test that model unloads properly and cleans up resources."""
    assert model.ready
    assert await model.unload()
    assert model._model is None
    assert model._output_names == []
    assert model._input_names == []


@pytest.mark.parametrize("fname", WELLKNOWN_MODEL_FILENAMES)
async def test_load_folder(fname, model_uri: str, model_settings: ModelSettings):
    """Test loading model from folder with wellknown filenames."""
    model_folder = os.path.dirname(model_uri)
    model_path = os.path.join(model_folder, fname)
    os.rename(model_uri, model_path)

    model_settings.parameters.uri = model_folder  # type: ignore

    model = OnnxModel(model_settings)
    model.ready = await model.load()

    assert model.ready
    assert model._model is not None


async def test_multi_output_model(multi_output_model: OnnxModel):
    """Test that multi-output model loads correctly."""
    assert multi_output_model.ready
    assert len(multi_output_model._output_names) == 2
    assert "output1" in multi_output_model._output_names
    assert "output2" in multi_output_model._output_names


async def test_load_through_symlinked_model_file(tmp_path):
    """Model .onnx file is a symlink — symlink-safe loader handles this."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real_model = real_dir / "model.onnx"
    onnx.save(_create_simple_onnx_model(), str(real_model))

    link_dir = tmp_path / "linked"
    link_dir.mkdir()
    link_model = link_dir / "model.onnx"
    link_model.symlink_to(real_model)

    settings = ModelSettings(
        name="symlinked-file-model",
        implementation=OnnxModel,
        parameters=ModelParameters(uri=str(link_model), version="v1.0.0"),
    )
    onnx_model = OnnxModel(settings)
    onnx_model.ready = await onnx_model.load()

    assert onnx_model.ready
    assert onnx_model._model is not None
    assert onnx_model._input_names == ["input"]


async def test_load_through_symlinked_directory(tmp_path):
    """Model dir is a symlink — symlink-safe loader handles this."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    onnx.save(_create_simple_onnx_model(), str(real_dir / "model.onnx"))

    link_dir = tmp_path / "link-dir"
    link_dir.symlink_to(real_dir)

    settings = ModelSettings(
        name="symlinked-dir-model",
        implementation=OnnxModel,
        parameters=ModelParameters(uri=str(link_dir / "model.onnx"), version="v1.0.0"),
    )
    onnx_model = OnnxModel(settings)
    onnx_model.ready = await onnx_model.load()

    assert onnx_model.ready
    assert onnx_model._model is not None
