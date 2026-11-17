import json

from asyncio import CancelledError
from enum import IntEnum
from pydantic import BaseModel, Field, ConfigDict
from typing import Any

from ..utils import generate_uuid
from ..settings import ModelSettings


class ModelUpdateType(IntEnum):
    Load = 1
    Unload = 2
    Cancel = 3


class Message(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
    )

    id: str = Field(default_factory=generate_uuid)


class ModelRequestMessage(Message):
    model_name: str
    model_version: str | None = None
    method_name: str
    method_args: list[Any] = []
    method_kwargs: dict[str, Any] = {}
    # Index of the frontend process that dispatched this request. Workers
    # use it to route the reply to the owning frontend's zmq endpoint.
    # Defaults to 0 (the only frontend when frontend_workers=1).
    frontend_id: int = 0


class ModelResponseMessage(Message):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    return_value: Any | None = None
    exception: Exception | CancelledError | None = None


class ModelUpdateMessage(Message):
    update_type: ModelUpdateType
    serialised_model_settings: str
    rollback: bool = False
    # For Cancel updates: the dispatched request id to evict from the
    # worker-side batcher queue.
    request_id: str | None = None

    def __init__(self, *args, **kwargs):
        model_settings = kwargs.pop("model_settings", None)
        if model_settings:
            as_dict = model_settings.model_dump(by_alias=True)
            # Ensure the private `_source` attr also gets serialised
            if model_settings._source:
                as_dict["_source"] = model_settings._source

            kwargs["serialised_model_settings"] = json.dumps(as_dict)

        return super().__init__(*args, **kwargs)

    @property
    def model_settings(self) -> ModelSettings:
        # Same dict-path deserialisation as deserialise_model_settings:
        # it restores the private `_source` attribute that pydantic's
        # JSON path (and the deprecated parse_raw) drops.
        return ModelSettings.model_validate(json.loads(self.serialised_model_settings))
