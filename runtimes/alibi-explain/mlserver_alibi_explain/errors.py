from mlserver.errors import MLServerError


class RemoteInferenceError(MLServerError):
    def __init__(self, code: int, reason: str):
        super().__init__(f"Remote inference call failed with {code}, {reason}")


class InvalidExplanationShape(MLServerError):
    def __init__(self, shape: list[int] | int):
        super().__init__(
            f"Expected a single element, but multiple were returned {shape}"
        )
