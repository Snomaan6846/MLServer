import os
import pytest
import asyncio
import platform

from unittest.mock import patch

from mlserver.utils import (
    _resolve_symlinks,
    get_model_uri,
    to_absolute_path,
    extract_headers,
    insert_headers,
    get_normalized_version,
    install_uvloop_event_loop,
)
from mlserver.version import __version__
from mlserver.types import InferenceRequest, InferenceResponse, Parameters
from mlserver.settings import ModelSettings, ModelParameters
from .fixtures import SumModel

test_get_model_uri_paramaters = [
    ("s3://bucket/key", None, "s3://bucket/key"),
    ("s3://bucket/key", "/mnt/models/model-settings.json", "s3://bucket/key"),
]
for scheme in ["", "file:"]:
    for uri, source, expected in [
        ("my-model.bin", None, "my-model.bin"),
        (
            "my-model.bin",
            "./my-model-folder/model-settings.json",
            "my-model-folder/my-model.bin",
        ),
        (
            "my-model.bin",
            "./my-model-folder/../model-settings.json",
            "my-model.bin",
        ),
        (
            "/an/absolute/path/my-model.bin",
            "/mnt/models/model-settings.json",
            "/an/absolute/path/my-model.bin",
        ),
    ]:
        test_get_model_uri_paramaters.append((scheme + uri, source, expected))


@pytest.mark.parametrize(
    "uri, source, expected",
    test_get_model_uri_paramaters,
)
async def test_get_model_uri(uri: str, source: str | None, expected: str):
    model_settings = ModelSettings(
        implementation=SumModel, parameters=ModelParameters(uri=uri)
    )
    model_settings._source = source
    with patch("os.path.isfile", return_value=True):
        model_uri = await get_model_uri(model_settings)

    assert model_uri == expected


@pytest.mark.parametrize(
    "parameters",
    [
        None,
        Parameters(),
        Parameters(headers={"foo": "bar2"}),
        Parameters(headers={"bar": "foo"}),
    ],
)
def test_insert_headers(parameters: Parameters):
    inference_request = InferenceRequest(inputs=[], parameters=parameters)
    headers = {"foo": "bar", "hello": "world"}
    insert_headers(inference_request, headers)

    assert inference_request.parameters is not None
    assert inference_request.parameters.headers == headers


@pytest.mark.parametrize(
    "parameters, expected",
    [
        (None, None),
        (Parameters(), None),
        (Parameters(headers={}), {}),
        (Parameters(headers={"foo": "bar"}), {"foo": "bar"}),
    ],
)
def test_extract_headers(parameters: Parameters, expected: dict[str, str]):
    inference_response = InferenceResponse(
        model_name="foo", outputs=[], parameters=parameters
    )
    headers = extract_headers(inference_response)

    assert headers == expected
    if inference_response.parameters:
        assert inference_response.parameters.headers is None


def _check_uvloop_availability():
    avail = True
    try:
        import uvloop  # noqa: F401
    except ImportError:  # pragma: no cover
        avail = False
    return avail


@pytest.mark.parametrize(
    "version, expected",
    [
        ("1.7.1+rhaiv.8", "1.7.1"),
        ("1.7.1", "1.7.1"),
        ("1.7.0.dev0", "1.7.0.dev0"),
    ],
)
def test_get_normalized_version(version: str | None, expected: str):
    assert get_normalized_version(version) == expected


def test_get_normalized_version_default_uses_current_version():
    assert get_normalized_version() == __version__.split("+", 1)[0]


class TestResolveSymlinks:
    """Tests for _resolve_symlinks used by to_absolute_path / get_model_uri."""

    def test_regular_path_unchanged(self, tmp_path):
        real_file = tmp_path / "model.onnx"
        real_file.touch()
        assert _resolve_symlinks(str(real_file)) == str(real_file)

    def test_relative_path_unchanged(self):
        assert _resolve_symlinks("some/relative/path") == os.path.join(
            "some", "relative", "path"
        )

    def test_resolves_file_symlink(self, tmp_path):
        real_file = tmp_path / "real-model.onnx"
        real_file.touch()
        link = tmp_path / "link-model.onnx"
        link.symlink_to(real_file)

        resolved = _resolve_symlinks(str(link))
        assert resolved == str(real_file)
        assert os.path.isfile(resolved)

    def test_resolves_directory_symlink(self, tmp_path):
        real_dir = tmp_path / "real-dir"
        real_dir.mkdir()
        (real_dir / "model.onnx").touch()
        link_dir = tmp_path / "link-dir"
        link_dir.symlink_to(real_dir)

        resolved = _resolve_symlinks(str(link_dir / "model.onnx"))
        assert resolved == str(real_dir / "model.onnx")
        assert os.path.isfile(resolved)

    def test_resolves_chained_symlinks(self, tmp_path):
        """Simulates KServe modelcar /proc/<pid>/root/ style chains.

        _resolve_symlinks resolves each component once.  outer -> mid
        is resolved at the outer component; the result (mid) is itself
        a symlink to actual, but it is only encountered as part of the
        *resolved* prefix — not as a new path component — so only one
        hop is followed per segment.  The important thing is the final
        path is valid and reachable.
        """
        real_dir = tmp_path / "actual"
        real_dir.mkdir()
        (real_dir / "model.onnx").touch()

        mid_link = tmp_path / "mid"
        mid_link.symlink_to(real_dir)
        outer_link = tmp_path / "outer"
        outer_link.symlink_to(mid_link)

        resolved = _resolve_symlinks(str(outer_link / "model.onnx"))
        # outer resolves to mid (which itself points to actual);
        # os.path.islink follows the first hop, giving us mid/model.onnx
        # which is reachable because mid -> actual.
        assert os.path.isfile(resolved)
        assert os.path.samefile(resolved, str(real_dir / "model.onnx"))

    def test_proc_path_not_resolved(self):
        """Paths under /proc/ are returned normalized but never resolved.

        /proc/<pid>/root is a procfs symlink into a container's root
        filesystem.  Resolving it would lose the proc-based access path.
        """
        assert _resolve_symlinks("/proc/5/root/models") == "/proc/5/root/models"
        assert (
            _resolve_symlinks("/proc/5/root/./models/../models/model.onnx")
            == "/proc/5/root/models/model.onnx"
        )

    def test_broken_symlink_keeps_original(self, tmp_path):
        link = tmp_path / "broken-link"
        link.symlink_to(tmp_path / "nonexistent-target")

        resolved = _resolve_symlinks(str(link))
        assert resolved == str(link)

    def test_relative_symlink_target(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        real_file = sub / "real.bin"
        real_file.touch()
        link = tmp_path / "rel-link"
        link.symlink_to(os.path.join("sub", "real.bin"))

        resolved = _resolve_symlinks(str(link))
        assert resolved == str(real_file)


class TestToAbsolutePathSymlinks:
    """Verify to_absolute_path resolves symlinks for all callers."""

    def test_resolves_symlinked_source_dir(self, tmp_path):
        real_dir = tmp_path / "real-models"
        real_dir.mkdir()
        (real_dir / "model.bin").touch()
        link_dir = tmp_path / "link-models"
        link_dir.symlink_to(real_dir)

        settings = ModelSettings(
            implementation=SumModel,
            parameters=ModelParameters(uri="model.bin"),
        )
        settings._source = str(link_dir / "model-settings.json")

        result = to_absolute_path(settings, "model.bin")
        assert result == str(real_dir / "model.bin")

    def test_no_source_resolves_symlinks(self, tmp_path):
        real_file = tmp_path / "real.bin"
        real_file.touch()
        link = tmp_path / "link.bin"
        link.symlink_to(real_file)

        settings = ModelSettings(
            implementation=SumModel,
            parameters=ModelParameters(uri=str(link)),
        )
        settings._source = None

        result = to_absolute_path(settings, str(link))
        assert result == str(real_file)


class TestGetModelUriSymlinks:
    """Integration tests: get_model_uri through symlinked paths."""

    async def test_symlinked_model_file(self, tmp_path):
        real_file = tmp_path / "real-model.onnx"
        real_file.touch()
        link = tmp_path / "link-model.onnx"
        link.symlink_to(real_file)

        settings = ModelSettings(
            implementation=SumModel,
            parameters=ModelParameters(uri=str(link)),
        )
        uri = await get_model_uri(settings)
        assert uri == str(real_file)

    async def test_symlinked_model_dir_with_wellknown(self, tmp_path):
        real_dir = tmp_path / "real-dir"
        real_dir.mkdir()
        (real_dir / "model.onnx").touch()
        link_dir = tmp_path / "link-dir"
        link_dir.symlink_to(real_dir)

        settings = ModelSettings(
            implementation=SumModel,
            parameters=ModelParameters(uri=str(link_dir)),
        )
        uri = await get_model_uri(settings, wellknown_filenames=["model.onnx"])
        assert uri == str(real_dir / "model.onnx")

    async def test_symlinked_dir_returns_resolved_folder(self, tmp_path):
        real_dir = tmp_path / "real-dir"
        real_dir.mkdir()
        link_dir = tmp_path / "link-dir"
        link_dir.symlink_to(real_dir)

        settings = ModelSettings(
            implementation=SumModel,
            parameters=ModelParameters(uri=str(link_dir)),
        )
        uri = await get_model_uri(settings)
        assert uri == str(real_dir)

    async def test_dir_no_wellknown_match_returns_folder(self, tmp_path):
        """URI points to a dir but no wellknown filename matches — returns dir."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "custom-name.bin").touch()

        settings = ModelSettings(
            implementation=SumModel,
            parameters=ModelParameters(uri=str(model_dir)),
        )
        uri = await get_model_uri(settings, wellknown_filenames=["model.onnx"])
        assert uri == str(model_dir)

    async def test_invalid_path_raises(self, tmp_path):
        """URI points to a path that is neither file nor directory."""
        from mlserver.errors import InvalidModelURI

        missing = str(tmp_path / "does-not-exist.bin")
        settings = ModelSettings(
            implementation=SumModel,
            parameters=ModelParameters(uri=missing),
        )
        with pytest.raises(InvalidModelURI):
            await get_model_uri(settings)

    async def test_proc_path_file(self, tmp_path):
        """/proc/ URI pointing to a file is preserved through normpath."""
        settings = ModelSettings(
            implementation=SumModel,
            parameters=ModelParameters(uri="/proc/5/root/models/model.onnx"),
        )
        with patch("os.path.isfile", return_value=True):
            uri = await get_model_uri(settings)
        assert uri == "/proc/5/root/models/model.onnx"

    async def test_proc_path_dir_with_wellknown(self, tmp_path):
        """/proc/ URI pointing to a dir finds wellknown file."""
        proc_uri = "/proc/5/root/models"

        def fake_isfile(path):
            return path == "/proc/5/root/models/model.onnx"

        def fake_isdir(path):
            return path == proc_uri

        settings = ModelSettings(
            implementation=SumModel,
            parameters=ModelParameters(uri=proc_uri),
        )
        with patch("os.path.isfile", side_effect=fake_isfile), patch(
            "os.path.isdir", side_effect=fake_isdir
        ):
            uri = await get_model_uri(settings, wellknown_filenames=["model.onnx"])
        assert uri == "/proc/5/root/models/model.onnx"

    async def test_symlinked_wellknown_file_in_dir(self, tmp_path):
        """Wellknown file inside a dir is a symlink — resolved correctly."""
        real_model = tmp_path / "real-model.onnx"
        real_model.touch()
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "model.onnx").symlink_to(real_model)

        settings = ModelSettings(
            implementation=SumModel,
            parameters=ModelParameters(uri=str(model_dir)),
        )
        uri = await get_model_uri(settings, wellknown_filenames=["model.onnx"])
        assert uri == str(real_model)


def test_uvloop_auto_install():
    uvloop_available = _check_uvloop_availability()
    install_uvloop_event_loop()
    policy = asyncio.get_event_loop_policy()

    if uvloop_available:
        assert type(policy).__module__.startswith("uvloop")
    else:
        if platform.system() == "Windows":
            assert isinstance(policy, asyncio.WindowsProactorEventLoopPolicy)
        elif platform.python_implementation() != "CPython":
            assert isinstance(policy, asyncio.DefaultEventLoopPolicy)
