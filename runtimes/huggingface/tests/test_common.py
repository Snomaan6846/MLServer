from unittest.mock import MagicMock, patch

import pytest
import torch
from optimum.onnxruntime.modeling_decoder import ORTModelForCausalLM
from transformers.models.gpt2 import GPT2ForQuestionAnswering

from mlserver.settings import ModelSettings, ModelParameters

from mlserver_huggingface.runtime import HuggingFaceRuntime
from mlserver_huggingface.settings import HuggingFaceSettings
from mlserver_huggingface.common import load_pipeline_from_settings


@pytest.mark.parametrize(
    "envs, expected",
    [
        ({"task": "translation", "task_suffix": "_en_to_fr"}, "translation_en_to_fr"),
        (
            {"task": "question-answering", "task_suffix": "_any_thing_else"},
            "question-answering",
        ),
    ],
)
def test_settings_task_name(envs: dict[str, str], expected: str):
    setting = HuggingFaceSettings.model_validate(envs)
    assert setting.task_name == expected


@pytest.mark.parametrize(
    "task, optimum_model, model, expected",
    [
        (
            "text-generation",
            True,
            "distilgpt2",
            ORTModelForCausalLM,
        ),
        (
            "question-answering",
            False,
            "distilgpt2",
            GPT2ForQuestionAnswering,
        ),
    ],
)
def test_load_pipeline(
    task: str,
    optimum_model: bool,
    model: str,
    expected: ORTModelForCausalLM | GPT2ForQuestionAnswering,
):
    hf_settings = HuggingFaceSettings(
        task=task,
        optimum_model=optimum_model,
        pretrained_model=model,
    )
    model_settings = ModelSettings(
        name="foo",
        implementation=HuggingFaceRuntime,
        parameters=ModelParameters(extra=hf_settings.model_dump()),
    )

    pipeline = load_pipeline_from_settings(hf_settings, model_settings)

    assert isinstance(pipeline.model, expected)


@pytest.mark.parametrize(
    "pretrained_model, parameters_uri, expected",
    [
        (None, None, None),
        (None, "", ""),
        (None, "/some/folder/model-artefacts", "/some/folder/model-artefacts"),
        ("", None, None),
        ("", "", ""),
        ("", "/some/folder/model-artefacts", "/some/folder/model-artefacts"),
        ("some-model", None, "some-model"),
        ("some-model", "", "some-model"),
        ("some-model", "/some/folder/model-artefacts", "some-model"),
        (
            "/some/other/folder/model-artefacts",
            None,
            "/some/other/folder/model-artefacts",
        ),
        (
            "/some/other/folder/model-artefacts",
            "",
            "/some/other/folder/model-artefacts",
        ),
        (
            "/some/other/folder/model-artefacts",
            "/some/folder/model-artefacts",
            "/some/other/folder/model-artefacts",
        ),
    ],
)
@patch("mlserver_huggingface.common._get_pipeline_class")
def test_pipeline_is_initialised_with_correct_model_param(
    mock_pipeline_factory,
    pretrained_model: str | None,
    parameters_uri: str | None,
    expected: str | None,
):
    mock_pipeline_factory.return_value = MagicMock()

    hf_settings = HuggingFaceSettings(pretrained_model=pretrained_model)
    model_params = ModelParameters(uri=parameters_uri)

    model_settings = ModelSettings(
        name="foo",
        implementation=HuggingFaceRuntime,
        parameters=model_params,
    )

    _ = load_pipeline_from_settings(hf_settings, model_settings)

    mock_pipeline_factory.return_value.assert_called_once()
    pipeline_call_args = mock_pipeline_factory.return_value.call_args

    assert pipeline_call_args.kwargs["model"] == expected


@pytest.mark.parametrize(
    "model_kwargs, expected",
    [
        (None, None),
        (
            {"load_in_8bit": True},
            {"load_in_8bit": True},
        ),
    ],
)
@patch("mlserver_huggingface.common._get_pipeline_class")
def test_pipeline_is_initialised_with_correct_model_kwargs(
    mock_pipeline_factory,
    model_kwargs: dict | None,
    expected: dict | None,
):
    mock_pipeline_factory.return_value = MagicMock()

    hf_settings = HuggingFaceSettings(model_kwargs=model_kwargs)
    model_params = ModelParameters(uri="dummy_uri")
    model_settings = ModelSettings(
        name="foo", implementation=HuggingFaceRuntime, parameters=model_params
    )
    _ = load_pipeline_from_settings(hf_settings, model_settings)

    mock_pipeline_factory.return_value.assert_called_once()
    pipeline_call_args = mock_pipeline_factory.return_value.call_args

    assert pipeline_call_args.kwargs["model_kwargs"] == expected


@pytest.mark.parametrize(
    "pretrained_model, model_kwargs, expected",
    [
        (
            "hf-internal-testing/tiny-bert-for-token-classification",
            {"torch_dtype": torch.float16},
            torch.float16,
        ),
        (
            "hf-internal-testing/tiny-bert-for-token-classification",
            None,
            torch.float32,
        ),
    ],
)
def test_pipeline_uses_model_kwargs(
    pretrained_model: str,
    model_kwargs: dict | None,
    expected: torch.dtype,
):
    hf_settings = HuggingFaceSettings(
        pretrained_model=pretrained_model,
        task="token-classification",
        model_kwargs=model_kwargs,
    )
    model_settings = ModelSettings(
        name="foo",
        implementation=HuggingFaceRuntime,
    )
    m = load_pipeline_from_settings(hf_settings, model_settings)

    assert m.model.dtype == expected


@pytest.mark.parametrize(
    "pretrained_model, device, expected",
    [
        (
            "hf-internal-testing/tiny-bert-for-token-classification",
            None,
            torch.device("cpu"),
        ),
        (
            "hf-internal-testing/tiny-bert-for-token-classification",
            -1,
            torch.device("cpu"),
        ),
        (
            "hf-internal-testing/tiny-bert-for-token-classification",
            "cpu",
            torch.device("cpu"),
        ),
    ],
)
def test_pipeline_cpu_device_set(
    pretrained_model: str,
    device: str | int | None,
    expected: torch.device,
):
    hf_settings = HuggingFaceSettings(
        pretrained_model=pretrained_model, task="token-classification", device=device
    )
    model_settings = ModelSettings(
        name="foo",
        implementation=HuggingFaceRuntime,
    )
    m = load_pipeline_from_settings(hf_settings, model_settings)

    assert m.model.device == expected


@pytest.mark.parametrize(
    "pretrained_model, task, input_batch_size, expected_batch_size",
    [
        (
            "hf-internal-testing/tiny-bert-for-token-classification",
            "token-classification",
            1,
            1,
        ),
        (
            "hf-internal-testing/tiny-bert-for-token-classification",
            "token-classification",
            0,
            1,
        ),
        (
            "hf-internal-testing/tiny-bert-for-token-classification",
            "token-classification",
            10,
            1,
        ),  # Neither pad_token nor eos_token defined revert to 1
    ],
)
def test_pipeline_checks_for_eos_and_pad_token(
    pretrained_model: str,
    task: str | None,
    input_batch_size: int | None,
    expected_batch_size: int | None,
):
    hf_settings = HuggingFaceSettings(pretrained_model=pretrained_model, task=task)
    model_params = ModelParameters()
    model_settings = ModelSettings(
        name="foo",
        implementation=HuggingFaceRuntime,
        parameters=model_params,
        max_batch_size=input_batch_size,
    )

    m = load_pipeline_from_settings(hf_settings, model_settings)

    assert m._batch_size == expected_batch_size
