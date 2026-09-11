from ..model import MLModel
from ..parallel.model import ParallelModel
from .adaptive import AdaptiveBatcher
from ..logging import logger


async def load_batching(model: MLModel) -> MLModel:
    if isinstance(model, ParallelModel):
        return model

    if model.settings.max_batch_size > 1 and model.settings.max_batch_time <= 0:
        logger.warning(
            "Setting max_batch_time equal to zero will result"
            " in batching having no effect, if you intend to "
            "use batching try setting it to a value > 0 for"
            " batching to take effect"
        )

    if model.settings.max_batch_size <= 1:
        return model

    if model.settings.max_batch_time <= 0:
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
