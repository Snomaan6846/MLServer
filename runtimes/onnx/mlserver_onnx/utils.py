import os
import pathlib
import logging
from typing import Any
from collections.abc import Iterable, Sequence

import onnx
from onnx import ModelProto, TensorProto, helper
import onnxruntime as ort

from mlserver.codecs.numpy import to_datatype
from mlserver.errors import ModelValidationError
from mlserver.types import Datatype, MetadataTensor

from .settings import OnnxSettings

logger = logging.getLogger(__name__)

PREDICT_OUTPUT = "predict"
VALID_OUTPUTS = [PREDICT_OUTPUT]

WELLKNOWN_MODEL_FILENAMES = ["model.onnx"]
DEFAULT_EXECUTION_PROVIDERS = ["CPUExecutionProvider"]
PROVIDERS_KEY = "providers"
PROVIDER_OPTIONS_KEY = "provider_options"
SESSION_OPTIONS_KEY = "session_options"
SESSION_CONFIG_ENTRIES_KEY = "session_config_entries"
RUN_OPTIONS_KEY = "run_options"


def _build_session_options(
    settings: OnnxSettings,
) -> ort.SessionOptions | None:
    """
    Build SessionOptions from settings.

    Args:
        settings: Parsed ONNX settings.

    Returns:
        SessionOptions or None if not configured.

    Raises:
        ModelValidationError: If session_options is invalid or unsupported.
    """
    session_options = settings.session_options
    if session_options is None:
        return None

    if not isinstance(session_options, dict):
        raise ModelValidationError(
            "OnnxModel session_options must be a dict of SessionOptions fields"
        )

    options = ort.SessionOptions()
    for key, value in session_options.items():
        if not hasattr(options, key):
            raise ModelValidationError(
                f"OnnxModel session option '{key}' is not supported"
            )
        setattr(options, key, value)

    return options


def _apply_session_config_entries(
    options: ort.SessionOptions | None, settings: OnnxSettings
) -> ort.SessionOptions | None:
    """
    Apply session_config_entries to SessionOptions.

    Args:
        options: Existing SessionOptions or None.
        settings: Parsed ONNX settings.

    Returns:
        SessionOptions with config entries applied.

    Raises:
        ModelValidationError: If session_config_entries is invalid.
    """
    entries = settings.session_config_entries
    if entries is None:
        return options

    if not isinstance(entries, dict):
        raise ModelValidationError(
            "OnnxModel session_config_entries must be a dict of string keys"
        )

    if options is None:
        options = ort.SessionOptions()

    for key, value in entries.items():
        if not isinstance(key, str):
            raise ModelValidationError(
                "OnnxModel session_config_entries keys must be strings"
            )
        options.add_session_config_entry(key, str(value))

    return options


def _build_run_options(settings: OnnxSettings) -> ort.RunOptions | None:
    """
    Build RunOptions from settings.

    Args:
        settings: Parsed ONNX settings.

    Returns:
        RunOptions or None if not configured.

    Raises:
        ModelValidationError: If run_options is invalid or unsupported.
    """
    run_options = settings.run_options
    if run_options is None:
        return None

    if not isinstance(run_options, dict):
        raise ModelValidationError(
            "OnnxModel run_options must be a dict of RunOptions fields"
        )

    options = ort.RunOptions()
    for key, value in run_options.items():
        if not hasattr(options, key):
            raise ModelValidationError(f"OnnxModel run option '{key}' is not supported")
        setattr(options, key, value)

    return options


def _get_providers(settings: OnnxSettings) -> list[str]:
    """
    Resolve execution providers from settings.

    Args:
        settings: Parsed ONNX settings.

    Returns:
        Ordered list of provider names.

    Raises:
        ModelValidationError: If providers is invalid.
    """
    providers = settings.providers
    if providers is None:
        return DEFAULT_EXECUTION_PROVIDERS
    if not isinstance(providers, list) or not providers:
        raise ModelValidationError(
            "OnnxModel providers must be a non-empty list of strings"
        )
    if not all(isinstance(provider, str) for provider in providers):
        raise ModelValidationError(
            "OnnxModel providers must be a non-empty list of strings"
        )

    return providers


def _get_provider_options(
    settings: OnnxSettings, providers: Sequence[str]
) -> list[dict[str, Any]] | None:
    """
    Resolve provider_options aligned with the providers list.

    Args:
        settings: Parsed ONNX settings.
        providers: Ordered list of provider names.

    Returns:
        List of dicts (one per provider) or None.

    Raises:
        ModelValidationError: If provider_options is invalid or length mismatches.
    """
    provider_options = settings.provider_options
    if provider_options is None:
        return None

    if isinstance(provider_options, dict):
        if len(providers) != 1:
            raise ModelValidationError(
                "OnnxModel provider_options dict requires a single provider"
            )
        return [provider_options]

    if not isinstance(provider_options, list):
        raise ModelValidationError(
            "OnnxModel provider_options must be a dict or list of dicts"
        )

    if not provider_options or not all(
        isinstance(option, dict) for option in provider_options
    ):
        raise ModelValidationError("OnnxModel provider_options must be a list of dicts")

    if len(provider_options) != len(providers):
        raise ModelValidationError(
            "OnnxModel provider_options must match providers length"
        )

    return provider_options


def _onnx_elem_type_to_datatype(elem_type: int) -> Datatype:
    """
    Map ONNX tensor element type to MLServer Datatype.

    Args:
        elem_type: ONNX tensor element type id.

    Returns:
        The MLServer datatype.

    Raises:
        ModelValidationError: If the element type is unsupported.
    """
    try:
        np_dtype = helper.tensor_dtype_to_np_dtype(elem_type)
        return to_datatype(np_dtype)
    except (KeyError, TypeError, ValueError):
        raise ModelValidationError(
            f"Unsupported ONNX tensor element type: {elem_type}"
        ) from None


def _onnx_shape_to_list(value_info: onnx.ValueInfoProto) -> list[int]:
    """
    Convert ONNX tensor shape to a list of ints; dynamic dims become -1.

    Args:
        value_info: ONNX ValueInfoProto.

    Returns:
        Shape as list of sizes (-1 for dynamic).
    """
    tensor_type = value_info.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.dim_value > 0 else -1)
    return dims


def _value_info_to_metadata(value_info: onnx.ValueInfoProto) -> MetadataTensor:
    """
    Convert ONNX ValueInfoProto to MetadataTensor.

    Args:
        value_info: ONNX ValueInfoProto.

    Returns:
        MetadataTensor with name, datatype, and shape.

    Raises:
        ModelValidationError: If type information is missing.
    """
    tensor_type = value_info.type.tensor_type
    if tensor_type is None or tensor_type.elem_type == 0:
        raise ModelValidationError(
            f"ONNX model tensor '{value_info.name}' missing type information"
        )

    return MetadataTensor(
        name=value_info.name,
        datatype=_onnx_elem_type_to_datatype(tensor_type.elem_type),
        shape=_onnx_shape_to_list(value_info),
    )


def _get_all_tensors(model: ModelProto) -> Iterable[TensorProto]:
    """Yield every TensorProto in the model, including nested subgraphs.

    Mirrors onnx.external_data_helper._get_all_tensors so that attribute
    tensors (e.g. Constant nodes) and tensors inside If/Loop/Scan
    subgraphs are not missed.
    """

    def _from_graph(graph):  # type: ignore[no-untyped-def]
        yield from graph.initializer
        for node in graph.node:
            for attr in node.attribute:
                if attr.HasField("t"):
                    yield attr.t
                yield from attr.tensors
                if attr.HasField("g"):
                    yield from _from_graph(attr.g)
                for sub_g in attr.graphs:
                    yield from _from_graph(sub_g)

    yield from _from_graph(model.graph)
    for func in model.functions:
        for node in func.node:
            for attr in node.attribute:
                if attr.HasField("t"):
                    yield attr.t
                yield from attr.tensors
                if attr.HasField("g"):
                    yield from _from_graph(attr.g)
                for sub_g in attr.graphs:
                    yield from _from_graph(sub_g)


def load_external_tensor_data(model: ModelProto, base_dir: str) -> None:
    """Load external tensor data using pure-Python I/O.

    ONNX >= 1.17 rejects symlinks in external data paths as a security
    measure (GHSA-538c-55jv-c5g9).  In KServe modelcar deployments the
    model directory is a symlink through /proc/<pid>/root/, so the
    built-in loader fails.

    This reads the data files with Python open(), which follows symlinks
    transparently, and attaches the raw bytes to each tensor.  The
    implementation mirrors onnx.external_data_helper.load_external_data_for_model
    but without the C++ symlink/hardlink checker.

    Note: unlike the native ONNX loader this does not validate the
    optional ``checksum`` field.  Data integrity should be ensured by
    the model distribution mechanism (e.g. container image layers).
    """
    loaded_count = 0
    for tensor in _get_all_tensors(model):
        if tensor.data_location != TensorProto.EXTERNAL:
            continue

        info = {entry.key: entry.value for entry in tensor.external_data}
        location = info.get("location")
        if not location:
            raise ValueError(
                f"Tensor {tensor.name!r} has data_location=EXTERNAL "
                f"but no 'location' key in external_data"
            )

        parts = pathlib.PurePosixPath(location).parts
        if os.path.isabs(location) or ".." in parts:
            raise ValueError(
                f"Unsafe external data location for tensor "
                f"{tensor.name!r}: {location!r}"
            )

        data_path = os.path.join(base_dir, location)
        offset = int(info["offset"]) if "offset" in info else None
        length = int(info["length"]) if "length" in info else None

        with open(data_path, "rb") as f:
            file_size = os.fstat(f.fileno()).st_size

            if offset is not None:
                if offset > file_size:
                    raise ValueError(
                        f"External data offset ({offset}) exceeds file size "
                        f"({file_size}) for tensor {tensor.name!r}"
                    )
                f.seek(offset)

            if length is not None:
                read_start = offset if offset is not None else 0
                available = file_size - read_start
                if length > available:
                    raise ValueError(
                        f"External data length ({length}) exceeds available "
                        f"data ({available} bytes from offset {read_start}) "
                        f"for tensor {tensor.name!r}"
                    )
                raw = f.read(length)
            else:
                raw = f.read()

        tensor.raw_data = raw
        tensor.data_location = TensorProto.DEFAULT
        del tensor.external_data[:]
        loaded_count += 1

    if loaded_count:
        logger.debug(
            "Loaded external data for %d tensor(s) from %s",
            loaded_count,
            base_dir,
        )


def load_onnx_model(model_uri: str) -> ModelProto:
    """Load an ONNX model with symlink-safe external data handling.

    Uses pure-Python I/O to load external tensor data, which follows
    symlinks transparently.  This avoids the symlink rejection added in
    ONNX >= 1.17 (GHSA-538c-55jv-c5g9) that breaks KServe modelcar
    deployments and shared-storage mounts with symlinked data files.
    """
    model = onnx.load(model_uri, load_external_data=False)
    base_dir = os.path.dirname(os.path.abspath(model_uri))
    load_external_tensor_data(model, base_dir)
    return model


def _extract_metadata(model_uri: str) -> dict[str, list[MetadataTensor]]:
    """
    Extract input and output metadata from the ONNX model file.

    Graph initializers (weights/constants) are excluded from inputs.

    Args:
        model_uri: Path to the ONNX model file.

    Returns:
        Dict with 'inputs' and 'outputs' lists of MetadataTensor.
    """
    model = onnx.load(model_uri, load_external_data=False)
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    inputs = [
        _value_info_to_metadata(value_info)
        for value_info in graph.input
        if value_info.name not in initializer_names
    ]
    outputs = [_value_info_to_metadata(value_info) for value_info in graph.output]

    return {"inputs": inputs, "outputs": outputs}
