from collections import defaultdict, OrderedDict
from collections.abc import Iterable
from typing import Any

import numpy as np

from ..codecs.lists import is_list_of
from ..codecs.string import decode_str
from ..codecs.numpy import (
    LOGICAL_BYTES_ELEMENTS,
    LOGICAL_BYTES_SENTINEL,
    _has_logical_elements,
    to_dtype,
)
from ..types import (
    InferenceRequest,
    InferenceResponse,
    Parameters,
    RequestInput,
    RequestOutput,
    ResponseOutput,
)
from .shape import Shape

# Adaptive batching merges BYTES inputs from potentially different
# transports.  Entries flagged as logical elements (the binary tensor
# transport) must not be reinterpreted as legacy packed buffers, so
# mixed representations are normalized to logical elements before the
# data merge.
_BYTES_DATATYPE = "BYTES"


def _bytes_mode(data: Any, request_input: RequestInput) -> str:
    """Classify a BYTES input's data representation."""
    if _has_logical_elements(request_input):
        return "logical"
    params = request_input.parameters
    content_type = getattr(params, "content_type", None) if params is not None else None
    if content_type == "str":
        # StringCodec contract: complete UTF-8 string elements
        return "string-codec"
    if is_list_of(data, str):
        return "string"
    if isinstance(data, bytes) or is_list_of(data, bytes):
        return "legacy"
    return "other"


def _standalone_elements(  # noqa: C901
    data: Any, request_input: RequestInput, mode: str
) -> list | None:
    """The input's elements exactly as its own codec decodes them standalone.

    Mixed BYTES batches merge these per-contract element lists, so every
    entry keeps the values (and types) it would have decoded to on its
    own: logical elements stay exact bytes, StringCodec elements become
    strings, legacy packed buffers split by the codec's itemsize with
    numpy's trailing-NUL stripping.
    """
    if mode == "logical":
        return list(data)
    if mode == "string-codec":
        if isinstance(data, bytes):
            data = [data]
        return [decode_str(elem) for elem in data]
    if mode == "string":
        return list(data)
    if mode == "legacy":
        if isinstance(data, bytes):
            data = [data]
        buffer = b"".join(data)
        itemsize = (
            request_input.shape[-1]
            if request_input.shape and request_input.shape[-1] > 0
            else 1
        )
        if len(buffer) % itemsize != 0:
            # Cannot split coherently — keep as-is and let the codec
            # surface the inconsistency
            return list(data)
        # numpy's fixed-width 'S' decode IS the semantics we want
        # (trailing-NUL stripping + itemsize slicing) — call it directly
        # instead of reimplementing (5.6x faster on large buffers)
        return np.frombuffer(buffer, dtype=f"S{itemsize}").tolist()
    return None


def _normalize_bytes_entries(
    entries: list[tuple[Any, RequestInput]],
) -> tuple[list[Any], bool]:
    """Normalize mixed BYTES representations to per-contract elements.

    Returns the (possibly unchanged) data list and whether normalization
    to logical elements occurred.  Each entry contributes the elements
    its own codec contract decodes standalone, so merged values (and
    their types) match the unbatched behaviour exactly.
    """
    # O(1) pre-check: only pay per-element classification when at least
    # one entry actually carries the logical-elements flag — restoring
    # the pre-change merge cost for non-binary-tensor traffic.
    if not any(_has_logical_elements(request_input) for _, request_input in entries):
        return [data for data, _ in entries], False

    modes = [_bytes_mode(data, request_input) for data, request_input in entries]
    if len(set(modes)) == 1:
        return [data for data, _ in entries], False

    normalized = []
    for (data, request_input), mode in zip(entries, modes):
        elements = _standalone_elements(data, request_input, mode)
        if elements is None:
            elements = list(data) if isinstance(data, list) else data
        normalized.append(elements)
    return normalized, True


def _get_data(payload: RequestInput | ResponseOutput):
    return getattr(payload.data, "root", payload.data)


def _get_parameters(payload: ResponseOutput) -> defaultdict[Any, Any]:
    parameters = defaultdict(list)
    if payload.parameters is not None:
        payload_parameters = payload.parameters.model_dump()
    for param_name, param_values in payload_parameters.items():
        if param_name in ["content_type", "headers"]:
            continue
        values = (
            [param_values]
            if isinstance(param_values, (str, bytes))
            or not isinstance(param_values, Iterable)
            else param_values
        )
        for param_value in values:
            parameters[param_name].append(param_value)
    if "content_type" in payload_parameters.keys():
        parameters["content_type"] = payload_parameters["content_type"]
    if "headers" in payload_parameters.keys():
        parameters["headers"] = payload_parameters["headers"]
    return parameters


def _merge_parameters(
    all_params: dict,
    parametrised_obj: (
        InferenceRequest | InferenceResponse | RequestInput | RequestOutput
    ),
) -> dict:
    if not parametrised_obj.parameters:
        return all_params

    obj_params = parametrised_obj.parameters.model_dump()
    return {**all_params, **obj_params}


def _merge_input_parameters(
    all_params: dict,
    parametrised_obj: (
        InferenceRequest | InferenceResponse | RequestInput | RequestOutput
    ),
) -> dict:
    if not parametrised_obj.parameters:
        return all_params
    obj_params = parametrised_obj.parameters.model_dump()
    if all_params == {}:
        return obj_params
    else:
        common_keys = set(all_params).intersection(set(obj_params)) - {
            "content_type",
            "headers",
        }
        uncommon_keys = set(all_params).union(set(obj_params)) - common_keys
        new_all_params = {}
        for key in common_keys:
            if isinstance(all_params[key], list):
                new_value = all_params[key] + [obj_params[key]]
                new_all_params[key] = new_value
            else:
                new_value = [all_params[key]]
                new_value.append(obj_params[key])
                new_all_params[key] = new_value
        for key in uncommon_keys:
            if key in all_params.keys():
                new_all_params[key] = all_params[key]
            if key in obj_params.keys():
                new_all_params[key] = obj_params[key]
    return new_all_params


def _merge_data(
    all_data: list | list[str] | list[bytes], dtype: "np.dtype | None" = None
) -> Any:
    sampled_datum = all_data[0]

    if isinstance(sampled_datum, str):
        return "".join(all_data)  # type: ignore

    if isinstance(sampled_datum, bytes):
        return b"".join(all_data)  # type: ignore

    # Check the WHOLE batch: a list-backed entry followed by an
    # ndarray-backed one (a JSON request batched with a binary one)
    # must concatenate, not sum — the list branch's sum() would
    # broadcast the list against the ndarray ([1,2] + array([3,4])
    # == [4,6]), silently producing the wrong element count for the
    # declared merged shape.
    if any(isinstance(datum, np.ndarray) for datum in all_data):
        # Ndarray-backed inputs (from the ndarray-native codecs)
        # concatenate their elements, matching the list path's sum()
        # flattening: minibatches of different sizes merge into the
        # flattened total that the merged shape declares. Returning
        # the arrays unmerged would hand the decoder a list of
        # unequal-length arrays it cannot convert.
        # Every entry converts to the tensor's DECLARED dtype: numpy's
        # promotion would otherwise combine e.g. a uint64 array with
        # an int64-converted list into float64, rounding large
        # integers before the decoder can cast them back.
        # Each entry flattens before concatenating: JSON requests may
        # carry nested data (shape [1,2] -> [[1,2]]) while
        # ndarray-native ones are flat, and concatenate() rejects the
        # dimension mismatch instead of merging them.
        if dtype is None:
            dtype = next(d.dtype for d in all_data if isinstance(d, np.ndarray))
        return np.concatenate(
            [np.asarray(datum, dtype=dtype).reshape(-1) for datum in all_data]
        )

    if isinstance(sampled_datum, list):
        return sum(all_data, [])

    # TODO: Should we raise an error if we couldn't merge the data?
    return all_data


class BatchedRequests:
    def __init__(self, inference_requests: dict[str, InferenceRequest] = {}):
        self.inference_requests = inference_requests

        # External IDs represent the incoming prediction IDs that need to match
        # 1:1 between request and response.
        # Since we can't ensure the uniqueness (or even presence) of the
        # external IDs, we'll also maintain our own list of internal IDs.
        self._ids_mapping: dict[str, str | None] = OrderedDict()

        # Minibatch here refers to the individual batch size of the input head
        # of each input request (i.e. the number of datapoints on each input
        # request)
        self._minibatch_sizes: dict[str, int] = OrderedDict()

        self.merged_request = self._merge_requests()

    def _merge_requests(self) -> InferenceRequest:
        inputs_index: dict[str, dict[str, RequestInput]] = defaultdict(OrderedDict)
        outputs_index: dict[str, dict[str, RequestOutput]] = defaultdict(OrderedDict)
        all_params: dict = {}
        has_outputs = False  # if no outputs are defined, then outputs=None

        for internal_id, inference_request in self.inference_requests.items():
            self._ids_mapping[internal_id] = inference_request.id
            all_params = _merge_parameters(all_params, inference_request)
            for request_input in inference_request.inputs:
                inputs_index[request_input.name][internal_id] = request_input

            if inference_request.outputs is not None:
                has_outputs = True
                for request_output in inference_request.outputs:
                    outputs_index[request_output.name][internal_id] = request_output

        inputs = [
            self._merge_request_inputs(request_inputs)
            for request_inputs in inputs_index.values()
        ]

        outputs = (
            [
                self._merge_request_outputs(request_outputs)
                for request_outputs in outputs_index.values()
            ]
            if has_outputs
            else None
        )

        # TODO: Should we add a 'fake' request ID?
        params = Parameters(**all_params) if all_params else None
        return InferenceRequest(inputs=inputs, outputs=outputs, parameters=params)

    def _merge_request_inputs(
        self, request_inputs: dict[str, RequestInput]
    ) -> RequestInput:
        # Note that minibatch sizes could be different on each input head,
        # however, to simplify the implementation, here we assume that it will
        # be the same across all of them
        batch_size = 0
        entries: list[tuple[Any, RequestInput]] = []
        all_params: dict = {}
        for internal_id, request_input in request_inputs.items():
            all_params = _merge_input_parameters(all_params, request_input)
            entries.append((_get_data(request_input), request_input))
            minibatch_shape = Shape(request_input.shape)
            self._minibatch_sizes[internal_id] = minibatch_shape.batch_size
            batch_size += minibatch_shape.batch_size

        sampled = next(iter(request_inputs.values()))
        normalized = False
        if str(sampled.datatype) == _BYTES_DATATYPE:
            # Mixed BYTES representations must share one codec mode
            # before the data merge.
            all_data, normalized = _normalize_bytes_entries(entries)
        else:
            all_data = [data for data, _ in entries]

        # The dtype-specific merge needs a valid datatype, but requests
        # assembled without pydantic validation (the gRPC converters use
        # InferenceRequest.construct) can carry an invalid datatype
        # string this far. On failure, re-run the public validation so
        # the error matches what the non-batched path would have
        # returned, instead of the internal enum message.
        try:
            dtype = to_dtype(sampled)
        except (KeyError, ValueError):
            RequestInput.model_validate(
                {
                    "name": sampled.name,
                    "shape": sampled.shape,
                    "datatype": sampled.datatype,
                    "data": _get_data(sampled),
                }
            )
            raise

        if normalized:
            # Normalized logical byte elements must never pass through
            # a fixed-width 'S' dtype (selected from the sampled entry
            # when it is a plain bytes ndarray): it truncates longer
            # elements before the logical-elements flag is applied.
            # Object dtype keeps every element's exact value and length.
            dtype = np.dtype(object)

        data = _merge_data(all_data, dtype=dtype)
        if normalized:
            all_params[LOGICAL_BYTES_ELEMENTS] = LOGICAL_BYTES_SENTINEL
            # A per-request content_type (e.g. "str") would select that
            # codec for the WHOLE merged batch and mis-decode the other
            # requests' normalized byte elements (a cross-request
            # failure).  Clear it so codec selection falls back to the
            # datatype — the NumPy codec honours the logical-elements
            # contract that _normalize_bytes_entries established.
            all_params.pop("content_type", None)
        parameters = Parameters(**all_params) if all_params else None

        # TODO: What should we do if list is empty?
        shape = Shape(sampled.shape)
        shape.batch_size = batch_size

        return RequestInput(
            name=sampled.name,
            datatype=sampled.datatype,
            shape=shape.to_list(),
            data=data,
            parameters=parameters,
        )

    def _merge_request_outputs(
        self, request_outputs: dict[str, RequestOutput]
    ) -> RequestOutput:
        all_params: dict = {}
        for internal_id, request_output in request_outputs.items():
            all_params = _merge_parameters(all_params, request_output)

        parameters = Parameters(**all_params) if all_params else None

        # TODO: What should we do if list is empty?
        sampled = next(iter(request_outputs.values()))

        return RequestOutput(name=sampled.name, parameters=parameters)

    def split_response(
        self, batched_response: InferenceResponse
    ) -> dict[str, InferenceResponse]:
        responses: dict[str, InferenceResponse] = {}

        for response_output in batched_response.outputs:
            response_outputs = self._split_response_output(response_output)

            for internal_id, response_output in response_outputs.items():
                if internal_id not in responses:
                    responses[internal_id] = InferenceResponse(
                        id=self._ids_mapping[internal_id],
                        model_name=batched_response.model_name,
                        model_version=batched_response.model_version,
                        outputs=[],
                        parameters=batched_response.parameters,
                    )

                responses[internal_id].outputs.append(response_output)

        return responses

    def _split_response_output(
        self, response_output: ResponseOutput
    ) -> dict[str, ResponseOutput]:
        all_data = self._split_data(response_output)
        if response_output.parameters is not None:
            all_parameters = self._split_parameters(response_output)
        else:
            all_parameters = None
        response_outputs = {}
        for internal_id, data in all_data.items():
            shape = Shape(response_output.shape)
            shape.batch_size = self._minibatch_sizes[internal_id]
            response_outputs[internal_id] = ResponseOutput(
                name=response_output.name,
                shape=shape.to_list(),
                data=data,
                datatype=response_output.datatype,
                parameters=(
                    all_parameters
                    if all_parameters is None
                    else all_parameters[internal_id]
                ),
            )

        return response_outputs

    def _split_data(self, response_output: ResponseOutput) -> dict[str, Any]:
        merged_shape = Shape(response_output.shape)
        element_size = merged_shape.elem_size
        merged_data = _get_data(response_output)
        idx = 0

        all_data = {}
        # TODO: Don't rely on array to have been flattened
        for internal_id, minibatch_size in self._minibatch_sizes.items():
            data = merged_data[idx : idx + minibatch_size * element_size]
            idx += minibatch_size * element_size
            all_data[internal_id] = data

        # NOTE: with an ndarray merged root these splits are views into
        # the merged buffer, not copies. Every current consumer is
        # read-only or copies (_encode_data's ravel-view guard); a hook
        # that ever mutates response data in place would corrupt the
        # sibling minibatches sharing this buffer.
        return all_data

    def _split_parameters(
        self, response_output: ResponseOutput
    ) -> dict[str, Parameters]:
        parameters = response_output.parameters
        if parameters is None:
            return {}
        if len(self._minibatch_sizes) == 1:
            # A single request's parameters were never batched, so preserve
            # their original values and types without flattening them.
            return {
                internal_id: parameters.model_copy(deep=True)
                for internal_id in self._minibatch_sizes
            }
        merged_parameters = _get_parameters(response_output)
        raw_parameters = parameters.model_dump()

        all_parameters = {}
        # Parameter lists have one entry per request, regardless of how many
        # data rows each request contains.
        for idx, internal_id in enumerate(self._minibatch_sizes):
            parameter_args = {}
            for parameter_name, parameter_values in merged_parameters.items():
                if parameter_name in ["content_type", "headers"]:
                    continue
                try:
                    raw_value = raw_parameters[parameter_name]
                    is_scalar = isinstance(raw_value, (str, bytes)) or not isinstance(
                        raw_value, Iterable
                    )
                    if is_scalar:
                        parameter_value = parameter_values[0]
                    else:
                        parameter_value = parameter_values[idx]
                    if parameter_value != []:
                        parameter_args[parameter_name] = (
                            parameter_value if is_scalar else str(parameter_value)
                        )
                except IndexError:
                    pass
            if "content_type" in merged_parameters.keys():
                parameter_args["content_type"] = merged_parameters["content_type"]
            if "headers" in merged_parameters.keys():
                parameter_args["headers"] = merged_parameters["headers"]
            parameter_obj = Parameters(**parameter_args)
            all_parameters[internal_id] = parameter_obj

        return all_parameters
