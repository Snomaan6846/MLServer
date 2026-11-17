import json
from typing import Any

from PIL import Image

from mlserver import MLModel
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    RequestInput,
    ResponseOutput,
    TensorData,
)
from mlserver.codecs import NumpyCodec, register_input_codec, DecodedParameterName
from mlserver.codecs.utils import InputOrOutput

_to_exclude = {
    "parameters": {DecodedParameterName},
    "inputs": {"__all__": {"parameters": {DecodedParameterName}}},
}


@register_input_codec
class PillowCodec(NumpyCodec):
    ContentType = "img"
    DefaultMode = "L"

    @classmethod
    def can_encode(cls, payload: Image.Image) -> bool:
        return isinstance(payload, Image.Image)

    @classmethod
    def _decode(cls, input_or_output: InputOrOutput) -> Image.Image:
        if input_or_output.datatype != "BYTES":
            # If not bytes, assume it's an array. The mode is inferred
            # from the array's dtype (e.g. uint8 -> "L", int32 -> "I");
            # forcing DefaultMode here would reject wider dtypes.
            image_array = super().decode_input(input_or_output)  # type: ignore
            return Image.fromarray(image_array)

        encoded = input_or_output.data
        # Unwrap the raw payload, which can be a str, bytes or a list
        # of either depending on how the request was sent.
        raw: Any = encoded.root if isinstance(encoded, TensorData) else encoded
        if isinstance(raw, str):
            raw = raw.encode()
        elif isinstance(raw, list):
            raw = b"".join(
                part.encode() if isinstance(part, str) else bytes(part) for part in raw
            )

        shape = input_or_output.shape
        return Image.frombytes(
            mode=cls.DefaultMode, size=(shape[0], shape[1]), data=raw
        )

    @classmethod
    def encode_output(  # type: ignore
        cls,
        name: str,
        payload: Image.Image,
    ) -> ResponseOutput:  # type: ignore
        # Raw pixel bytes in the image's mode, the counterpart of
        # Image.frombytes() in _decode().
        return ResponseOutput(
            name=name, shape=payload.size, datatype="BYTES", data=payload.tobytes()
        )

    @classmethod
    def decode_output(  # type: ignore
        cls, response_output: ResponseOutput
    ) -> Image.Image:
        return cls._decode(response_output)

    @classmethod
    def encode_input(  # type: ignore
        cls,
        name: str,
        payload: Image.Image,
    ) -> RequestInput:  # type: ignore
        output = cls.encode_output(name, payload)
        return RequestInput(
            name=output.name,
            shape=output.shape,
            datatype=output.datatype,
            data=output.data,
        )

    @classmethod
    def decode_input(cls, request_input: RequestInput) -> Image.Image:  # type: ignore
        return cls._decode(request_input)


class EchoRuntime(MLModel):
    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        outputs = []
        for request_input in payload.inputs:
            decoded_input = self.decode(request_input)
            print(f"------ Encoded Input ({request_input.name}) ------")
            as_dict = request_input.dict(exclude=_to_exclude)  # type: ignore
            print(json.dumps(as_dict, indent=2))
            print(f"------ Decoded input ({request_input.name}) ------")
            print(decoded_input)

            outputs.append(
                ResponseOutput(
                    name=request_input.name,
                    datatype=request_input.datatype,
                    shape=request_input.shape,
                    data=request_input.data,
                )
            )

        return InferenceResponse(model_name=self.name, outputs=outputs)
