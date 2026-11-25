import asyncio
import random
import string
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List

import numpy as np
from fastapi import Body

try:
    # NOTE: This is used in the EnvModel down below, which tests dynamic
    # loading custom environments.
    # Therefore, it is expected (and alright) that this package is not present
    # some times.
    import sklearn
except ImportError:
    sklearn = None

from collections.abc import AsyncIterator

from mlserver import MLModel
from mlserver.types import (
    Datatype,
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
    Parameters,
    TensorData,
)
from mlserver.codecs import NumpyCodec, decode_args, StringCodec
from mlserver.handlers.custom import custom_handler
from mlserver.errors import MLServerError


class DeadlineModel(MLModel):
    """A model whose predict always outlives a short inference deadline.

    Distinct from ``SlowModel`` (whose *load* sleeps, for readiness
    tests): this one loads instantly and sleeps in predict, for the
    end-to-end inference-timeout test.
    """

    async def load(self) -> bool:
        self.ready = True
        return True

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        await asyncio.sleep(2)
        raise RuntimeError("unreachable under a short inference_timeout")


class SumModel(MLModel):
    @custom_handler(rest_path="/my-custom-endpoint")
    async def my_payload(self, payload: list = Body(...)) -> int:
        return sum(payload)

    @custom_handler(rest_path="/custom-endpoint-with-long-response")
    async def long_response_endpoint(self, length: int = Body(...)) -> dict[str, str]:
        alphabet = string.ascii_lowercase
        response = "".join(random.choice(alphabet) for i in range(length))
        return {"foo": response}

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        decoded = self.decode(payload.inputs[0])
        total = decoded.sum(axis=1, keepdims=True)

        output = NumpyCodec.encode_output(name="total", payload=total)
        response = InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[output],
        )

        if payload.parameters and payload.parameters.headers:
            # "Echo" headers back prefixed by `x-`
            request_headers = payload.parameters.headers
            response_headers = {}
            for header_name, header_value in request_headers.items():
                if header_name.startswith("x-"):
                    response_headers[header_name] = header_value

            response.parameters = Parameters(headers=response_headers)

        return response


class TextModel(MLModel):
    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        text = StringCodec.decode_input(payload.inputs[0])[0]
        return InferenceResponse(
            model_name=self._settings.name,
            outputs=[
                StringCodec.encode_output(
                    name="output",
                    payload=[text],
                    use_bytes=True,
                ),
            ],
        )


class TextStreamModel(MLModel):
    async def predict_stream(
        self, payloads: AsyncIterator[InferenceRequest]
    ) -> AsyncIterator[InferenceResponse]:
        payload = [_ async for _ in payloads][0]
        text = StringCodec.decode_input(payload.inputs[0])[0]
        words = text.split(" ")

        split_text = []
        for i, word in enumerate(words):
            split_text.append(word if i == 0 else " " + word)

        for word in split_text:
            await asyncio.sleep(0.5)
            yield InferenceResponse(
                model_name=self._settings.name,
                outputs=[
                    StringCodec.encode_output(
                        name="output",
                        payload=[word],
                        use_bytes=True,
                    ),
                ],
            )


class ErrorModel(MLModel):
    error_message = "something really bad happened"

    async def load(self) -> bool:
        if self._settings.parameters:
            load_error = getattr(self._settings.parameters, "load_error", False)
            if load_error:
                raise MLServerError(self.error_message)

            load_returns_false = getattr(
                self._settings.parameters, "load_returns_false", False
            )
            if load_returns_false:
                return False

        return True

    async def unload(self) -> bool:
        if self._settings.parameters:
            unload_error = getattr(self._settings.parameters, "unload_error", False)
            if unload_error:
                raise MLServerError(self.error_message)

            unload_returns_false = getattr(
                self._settings.parameters, "unload_returns_false", False
            )
            if unload_returns_false:
                return False

        return True

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        raise MLServerError(self.error_message)


class SlowModel(MLModel):
    async def load(self) -> bool:
        await asyncio.sleep(10)
        return True

    async def infer(self, payload: InferenceRequest) -> InferenceResponse:
        await asyncio.sleep(10)
        return InferenceResponse(id=payload.id, model_name=self.name, outputs=[])


class SimpleModel(MLModel):
    @decode_args
    async def predict(self, foo: np.ndarray, bar: List[str]) -> np.ndarray:
        return foo.sum(axis=1, keepdims=True)


class EnvModel(MLModel):
    async def load(self):
        self._sklearn_version = sklearn.__version__
        return True

    async def predict(self, inference_request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            model_name=self.name,
            outputs=[
                StringCodec.encode_output("sklearn_version", [self._sklearn_version]),
            ],
        )


class EchoModel(MLModel):
    async def load(self) -> bool:
        print("Echo Model Initialized")
        return await super().load()

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[
                ResponseOutput(
                    name=input.name,
                    shape=input.shape,
                    datatype=input.datatype,
                    data=input.data,
                    parameters=input.parameters,
                )
                for input in payload.inputs
            ],
        )


class FailingStreamModel(SumModel):
    """A streaming model that fails on demand: before the first
    event (``fail_before_first``, set on the loaded instance) or
    after one successful event (the default). Inherits predict()
    from SumModel so the first event succeeds.

    The optional ``stream_closed`` hook (an asyncio.Event set on the
    loaded instance) pins that the SSE settlement closes the model's
    iterator on the error path too — the same cleanup a normal
    completion runs."""

    async def predict_stream(self, payloads):
        try:
            if getattr(self, "fail_before_first", False):
                raise MLServerError("mid-stream failure")
            async for payload in payloads:
                yield await self.predict(payload)
                raise MLServerError("mid-stream failure")
        finally:
            closed = getattr(self, "stream_closed", None)
            if closed is not None:
                closed.set()


class TrackedSingleShotModel(SumModel):
    """A single-shot model whose predict cleanup is observable: pins
    the single-shot SSE settlement (the model's async finally must
    complete, never be interrupted by a second cancellation). The
    test sets the cleanup hooks on the loaded instance."""

    async def predict(self, payload):
        entered = getattr(self, "cleanup_entered", None)
        release = getattr(self, "slow_release", None)
        gate = getattr(self, "predict_gate", None)
        try:
            if gate is not None:
                # Hold the inference open (the test cancels the
                # request while it runs): the settlement's cancel
                # lands here, INSIDE the try, so the finally below
                # always runs.
                await gate.wait()
            return await super().predict(payload)
        finally:
            if entered is not None:
                entered.set()
            if release is not None:
                await release.wait()
            closed = getattr(self, "stream_closed", None)
            if closed is not None:
                closed.set()


class TrackedStreamModel(SumModel):
    """A streaming model whose predict_stream closure is observable:
    pins the SSE response's cleanup ownership — the model iterator
    must close when the ASGI response exits, on every exit path
    (completion, send failure, cancellation). The test sets a
    ``stream_closed`` event on the loaded instance."""

    async def predict_stream(self, payloads):
        try:
            async for payload in payloads:
                for _ in range(getattr(self, "stream_length", 6)):
                    yield await self.predict(payload)
        finally:
            # Optional async cleanup hooks (set on the loaded instance
            # by the test): the model's finally may need to AWAIT
            # resource release — a WELL-BEHAVED cleanup survives a
            # cancellation delivered while it waits (the settlement's
            # once-only cancel) and runs to completion.
            entered = getattr(self, "cleanup_entered", None)
            if entered is not None:
                entered.set()
            release = getattr(self, "slow_release", None)
            if release is not None:
                inner = asyncio.ensure_future(release.wait())
                try:
                    await asyncio.shield(inner)
                except asyncio.CancelledError:
                    await inner
            closed = getattr(self, "stream_closed", None)
            if closed is not None:
                closed.set()


class DatetimeEchoModel(EchoModel):
    """Echoes the input with datetime objects in the BYTES output
    data: pins the wire_json_scalars conversion of tensor DATA in the
    pre-encoded REST path (a UTC datetime must render with Z on
    /infer, as the base's response-model serialization did — and keep
    orjson's +00:00 on /infer_stream, whose ServerSentEvent encoding
    is unchanged from the base)."""

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        response = await super().predict(payload)
        response.outputs[0].datatype = Datatype.BYTES
        response.outputs[0].data = TensorData(
            root=[
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=2))),
            ]
        )
        return response


class ExoticEchoModel(EchoModel):
    """Echoes the input with the full exotic-value mix in the BYTES
    output data (UTC datetime, offset datetime, Decimal, timedelta,
    loose numpy scalar): pins that the frontend and the worker render
    the SAME response byte-identically — one serializer, no
    variations between serving paths."""

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        response = await super().predict(payload)
        response.outputs[0].datatype = Datatype.BYTES
        response.outputs[0].data = TensorData(
            root=[
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=2))),
                Decimal("3.14"),
                timedelta(seconds=2),
                np.float32(0.1),
            ]
        )
        return response


class MetadataEchoModel(EchoModel):
    """Echoes the input and attaches exotic-typed response metadata
    (Decimal, timedelta): pins the JSON-mode scalar conversion of the
    response parameters in the pre-encoded REST path."""

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        response = await super().predict(payload)
        # Extra-allow models take non-schema fields directly: they
        # render FLAT in the dumped parameters, matching the base's
        # response-model serialization.
        response.parameters = Parameters(
            decimal=Decimal("3.14"), delta=timedelta(seconds=2)
        )
        response.outputs[0].parameters = Parameters(decimal=Decimal("2.72"))
        return response
