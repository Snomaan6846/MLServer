import pytest
import asyncio

from mlserver.errors import ModelNotFound, MLServerError
from mlserver.registry import MultiModelRegistry
from mlserver.handlers import ModelRepositoryHandlers
from mlserver.settings import ModelSettings, ModelParameters
from mlserver.types import RepositoryIndexRequest, State


async def test_index(
    model_repository_handlers: ModelRepositoryHandlers,
    repository_index_request: RepositoryIndexRequest,
    sum_model_settings: ModelSettings,
):
    repo_index = list(await model_repository_handlers.index(repository_index_request))

    assert len(repo_index) == 1
    assert repo_index[0].name == sum_model_settings.name
    assert sum_model_settings.parameters is not None
    assert repo_index[0].version == sum_model_settings.parameters.version
    assert repo_index[0].state == State.READY


async def test_index_unavailable_model(
    model_repository_handlers: ModelRepositoryHandlers,
    repository_index_request: RepositoryIndexRequest,
    sum_model_settings: ModelSettings,
):
    await model_repository_handlers.unload(sum_model_settings.name)
    repo_index = list(await model_repository_handlers.index(repository_index_request))

    assert len(repo_index) == 1
    assert repo_index[0].name == sum_model_settings.name
    assert sum_model_settings.parameters is not None
    assert repo_index[0].version == sum_model_settings.parameters.version
    assert repo_index[0].state == State.UNAVAILABLE


@pytest.mark.parametrize("ready,expected", [(None, 1), (True, 0), (False, 1)])
async def test_index_filter_ready(
    model_repository_handlers: ModelRepositoryHandlers,
    repository_index_request: RepositoryIndexRequest,
    sum_model_settings: ModelSettings,
    ready: bool | None,
    expected: int,
):
    await model_repository_handlers.unload(sum_model_settings.name)

    repository_index_request.ready = ready
    repo_index = list(await model_repository_handlers.index(repository_index_request))

    assert len(repo_index) == expected


async def test_unload(
    model_repository_handlers: ModelRepositoryHandlers,
    model_registry: MultiModelRegistry,
    sum_model_settings: ModelSettings,
):
    await model_repository_handlers.unload(sum_model_settings.name)

    with pytest.raises(ModelNotFound):
        await model_registry.get_model(sum_model_settings.name)


async def test_unload_not_found(
    model_repository_handlers: ModelRepositoryHandlers,
):
    with pytest.raises(ModelNotFound):
        await model_repository_handlers.unload("not-existing")


async def test_load_not_found(
    model_repository_handlers: ModelRepositoryHandlers,
):
    with pytest.raises(ModelNotFound):
        await model_repository_handlers.load("not-existing")


async def test_load_removes_stale_models(
    model_repository_handlers: ModelRepositoryHandlers,
    repository_index_request: RepositoryIndexRequest,
    model_registry: MultiModelRegistry,
    sum_model_settings: ModelSettings,
):
    # Load a few models which are not present on the repository (including a
    # default one), therefore they will be stale
    stale_settings = sum_model_settings.copy(deep=True)
    assert stale_settings.parameters is not None
    stale_settings.parameters.version = None
    await model_registry.load(stale_settings)

    to_load = ["v0", "v1", "v2"]
    for version in to_load:
        stale_settings = sum_model_settings.copy(deep=True)
        assert stale_settings.parameters is not None
        stale_settings.parameters.version = version
        await model_registry.load(stale_settings)

    # Validate that the stale test models have been loaded
    registry_models = await model_registry.get_models(sum_model_settings.name)
    stale_length = (
        len(to_load)
        + 1  # Count the (stale) default model
        + 1  # Count the previous (non-stale) model
    )
    assert len(registry_models) == stale_length

    # Reload our model and validate whether stale models have been removed
    await model_repository_handlers.load(sum_model_settings.name)

    # Assert that stale models have been removed from both the registry (and
    # ensure they are not present on the repository either)
    registry_models = await model_registry.get_models(sum_model_settings.name)
    repo_models = list(await model_repository_handlers.index(repository_index_request))
    assert sum_model_settings.parameters is not None
    expected_version = sum_model_settings.parameters.version

    assert len(registry_models) == 1
    assert registry_models[0].version == expected_version

    assert len(repo_models) == 1
    assert repo_models[0].version == expected_version


async def test_load_stale_cleanup_attempts_all_with_failures(
    model_repository_handlers: ModelRepositoryHandlers,
    model_registry: MultiModelRegistry,
    sum_model_settings: ModelSettings,
    mocker,
):
    """Test repository load cleanup uses return_exceptions
    and attempts all stale versions"""
    # Load 3 stale versions in registry
    stale_versions = ["v0", "v1", "v2"]
    for version in stale_versions:
        stale_settings = sum_model_settings.copy(deep=True)
        assert stale_settings.parameters is not None
        stale_settings.parameters.version = version
        await model_registry.load(stale_settings)

    # Mock v1 to fail unload
    models = await model_registry.get_models(sum_model_settings.name)
    v1_model = [m for m in models if m.version == "v1"][0]
    mocker.patch.object(v1_model, "unload", side_effect=MLServerError("Unload failed"))

    # Reload model from repository (v1.2.3) - should attempt
    # to cleanup all stale versions
    from mlserver.errors import ModelUnloadError

    with pytest.raises(ModelUnloadError, match="Failed to cleanup 1 of 3 stale"):
        await model_repository_handlers.load(sum_model_settings.name)

    # v1.2.3 should be loaded successfully
    loaded_model = await model_registry.get_model(sum_model_settings.name, "v1.2.3")
    assert loaded_model.ready

    # All stale models removed from registry (unregister happens before unload)
    all_models = await model_registry.get_models(sum_model_settings.name)
    assert len(all_models) == 1
    assert all_models[0].version == "v1.2.3"


async def test_union_model_settings_deduplication_and_precedence(
    model_repository_handlers: ModelRepositoryHandlers,
    sum_model_settings: ModelSettings,
):
    """Test _union_model_settings deduplicates by
    (name, version) and registry takes precedence"""
    from ..fixtures import SumModel

    repository_settings = [
        ModelSettings(
            name="model-a",
            implementation=SumModel,
            parameters=ModelParameters(version="v1", uri="repo-uri"),
        ),
        ModelSettings(
            name="model-b",
            implementation=SumModel,
            parameters=ModelParameters(version="v2"),
        ),
    ]

    registry_settings = [
        ModelSettings(
            name="model-a",
            implementation=SumModel,
            parameters=ModelParameters(version="v1", uri="registry-uri"),
        ),
        ModelSettings(
            name="model-c",
            implementation=SumModel,
            parameters=ModelParameters(version="v3"),
        ),
    ]

    result = model_repository_handlers._union_model_settings(
        repository_settings, registry_settings
    )

    # Should have 3 models: A v1, B v2, C v3
    assert len(result) == 3

    model_map = {
        (m.name, m.parameters.version if m.parameters else None): m for m in result
    }  # type: ignore

    # Model A v1 should use registry settings (not repository)
    assert model_map[("model-a", "v1")].parameters.uri == "registry-uri"  # type: ignore

    # Model B and C should be present
    assert ("model-b", "v2") in model_map
    assert ("model-c", "v3") in model_map


async def test_get_state_returns_loading_for_not_ready_model(
    model_repository_handlers: ModelRepositoryHandlers,
    model_registry: MultiModelRegistry,
):
    """Test _get_state returns LOADING state for models with ready=False"""
    from ..fixtures import SlowModel

    # Load a slow model that takes time
    slow_settings = ModelSettings(name="slow-model", implementation=SlowModel)

    # Start loading in background
    load_task = asyncio.create_task(model_registry.load(slow_settings))
    await asyncio.sleep(0.1)  # Let it start loading

    # Query state while loading
    state = await model_repository_handlers._get_state(slow_settings)
    assert state == State.LOADING

    # Wait for load to complete
    await load_task

    # Now should be READY
    state = await model_repository_handlers._get_state(slow_settings)
    assert state == State.READY

    # Unload model
    await model_registry.unload("slow-model")

    # Should be UNAVAILABLE
    state = await model_repository_handlers._get_state(slow_settings)
    assert state == State.UNAVAILABLE
