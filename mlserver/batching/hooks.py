from ..model import MLModel
from .adaptive import AdaptiveBatcher
from ..logging import logger


def warn_batching_misconfigured(model: MLModel) -> None:
    """Warn when a model's batching settings cannot take effect."""
    if model.settings.max_batch_size > 1 and model.settings.max_batch_time <= 0:
        logger.warning(
            "Setting max_batch_time equal to zero will result"
            " in batching having no effect, if you intend to "
            "use batching try setting it to a value > 0 for"
            " batching to take effect"
        )


async def load_batching(model: MLModel) -> MLModel:
    warn_batching_misconfigured(model)

    if model.settings.max_batch_size <= 1:
        return model

    if model.settings.max_batch_time <= 0:
        return model

    if AdaptiveBatcher.get_batcher(model) is not None:
        # Idempotent per instance: a superseded origin's hook chain
        # re-runs on the wrapper an apply already batched (the
        # stale-publication swap). Wrapping again would nest the
        # batching decorators and orphan the first batcher.
        return model

    batcher = AdaptiveBatcher(model)
    batcher.setup()

    return model


async def unload_batching(model: MLModel) -> MLModel:
    """Clean up batching resources when unloading a model."""
    batcher = AdaptiveBatcher.get_batcher(model)
    if not batcher:
        return model

    await batcher.teardown()

    return model
