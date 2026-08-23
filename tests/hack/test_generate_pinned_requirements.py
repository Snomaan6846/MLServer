"""Tests for hack/generate-pinned-requirements.py."""

import importlib.util
import json
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "generate_pinned_requirements",
    _REPO_ROOT / "hack" / "generate-pinned-requirements.py",
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

resolve_platform = _mod.resolve_platform
resolve_platform_groups = _mod.resolve_platform_groups
resolve_download_timeout = _mod.resolve_download_timeout
load_config = _mod.load_config
build_requirement_strings = _mod.build_requirement_strings
_kill_registered_procs = _mod._kill_registered_procs
run_pip_command = _mod.run_pip_command
normalize_distribution_name = _mod.normalize_distribution_name
redact_index_url = _mod.redact_index_url
_name_version_from_filename = _mod._name_version_from_filename
get_base_image_from_dockerfile = _mod.get_base_image_from_dockerfile
PLATFORM_TAGS = _mod.PLATFORM_TAGS
PLATFORM_ALIASES = _mod.PLATFORM_ALIASES
DEFAULT_PLATFORM = _mod.DEFAULT_PLATFORM
SKIP_NOT_AVAILABLE = _mod.SKIP_NOT_AVAILABLE


class TestResolvePlatform:
    def test_canonical_x86_64(self):
        result = resolve_platform("x86_64")
        assert result == [
            "manylinux2014_x86_64",
            "manylinux_2_34_x86_64",
            "linux_x86_64",
        ]

    def test_canonical_aarch64(self):
        result = resolve_platform("aarch64")
        assert result == [
            "manylinux2014_aarch64",
            "manylinux_2_34_aarch64",
            "linux_aarch64",
        ]

    def test_canonical_ppc64le(self):
        result = resolve_platform("ppc64le")
        assert result == [
            "manylinux2014_ppc64le",
            "manylinux_2_34_ppc64le",
            "linux_ppc64le",
        ]

    def test_canonical_s390x(self):
        result = resolve_platform("s390x")
        assert result == [
            "manylinux2014_s390x",
            "manylinux_2_34_s390x",
            "linux_s390x",
        ]

    def test_alias_amd64(self):
        assert resolve_platform("amd64") == resolve_platform("x86_64")

    def test_alias_arm64(self):
        assert resolve_platform("arm64") == resolve_platform("aarch64")

    def test_invalid_arch_raises(self):
        with pytest.raises(ValueError, match="Unsupported architecture 'mips64'"):
            resolve_platform("mips64")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Unsupported architecture"):
            resolve_platform("")

    def test_error_message_lists_supported(self):
        with pytest.raises(ValueError) as exc_info:
            resolve_platform("invalid")
        msg = str(exc_info.value)
        for name in ["x86_64", "aarch64", "ppc64le", "s390x", "amd64", "arm64"]:
            assert name in msg

    def test_case_sensitivity(self):
        with pytest.raises(ValueError):
            resolve_platform("X86_64")
        with pytest.raises(ValueError):
            resolve_platform("AMD64")

    def test_return_type_is_list_of_strings(self):
        result = resolve_platform("x86_64")
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)

    def test_all_canonical_names_in_platform_tags(self):
        for name in PLATFORM_TAGS:
            result = resolve_platform(name)
            assert result == PLATFORM_TAGS[name]

    def test_all_aliases_resolve(self):
        for alias, canonical in PLATFORM_ALIASES.items():
            result = resolve_platform(alias)
            assert result == PLATFORM_TAGS[canonical]


class TestLoadConfigPlatformValidation:
    def _write_config(self, tmp_path, variants):
        config = {"variants": variants}
        config_path = tmp_path / "requirements-config.json"
        config_path.write_text(json.dumps(config))
        return tmp_path

    def test_valid_platforms_field(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cpu",
                    "dockerfile": "Dockerfile",
                    "platforms": ["x86_64", "aarch64"],
                }
            ],
        )
        result = load_config(d)
        assert result["variants"][0]["platforms"] == ["x86_64", "aarch64"]

    def test_platforms_with_aliases(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cpu",
                    "dockerfile": "Dockerfile",
                    "platforms": ["amd64", "arm64"],
                }
            ],
        )
        result = load_config(d)
        assert result["variants"][0]["platforms"] == ["amd64", "arm64"]

    def test_invalid_arch_in_platforms_raises(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cpu",
                    "dockerfile": "Dockerfile",
                    "platforms": ["x86_64", "mips64"],
                }
            ],
        )
        with pytest.raises(ValueError, match="Unsupported architecture 'mips64'"):
            load_config(d)

    def test_empty_platforms_list_raises(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cpu",
                    "dockerfile": "Dockerfile",
                    "platforms": [],
                }
            ],
        )
        with pytest.raises(ValueError, match="non-empty list"):
            load_config(d)

    def test_platforms_not_a_list_raises(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cpu",
                    "dockerfile": "Dockerfile",
                    "platforms": "x86_64",
                }
            ],
        )
        with pytest.raises(ValueError, match="non-empty list"):
            load_config(d)

    def test_platforms_omitted_is_valid(self, tmp_path):
        d = self._write_config(tmp_path, [{"name": "cpu", "dockerfile": "Dockerfile"}])
        result = load_config(d)
        assert "platforms" not in result["variants"][0]

    def test_cuda_variant_with_ppc64le_warns(self, tmp_path, capsys):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cuda",
                    "dockerfile": "Dockerfile.cuda",
                    "platforms": ["x86_64", "ppc64le"],
                }
            ],
        )
        load_config(d)
        captured = capsys.readouterr()
        assert "ppc64le" in captured.err
        assert "CUDA" in captured.err

    def test_cuda_variant_with_ppc64le_does_not_raise(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cuda",
                    "dockerfile": "Dockerfile.cuda",
                    "platforms": ["x86_64", "ppc64le"],
                }
            ],
        )
        load_config(d)

    def test_duplicate_platforms_allowed(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cpu",
                    "dockerfile": "Dockerfile",
                    "platforms": ["x86_64", "x86_64"],
                }
            ],
        )
        result = load_config(d)
        assert result["variants"][0]["platforms"] == ["x86_64", "x86_64"]

    def test_missing_required_fields_still_raises(self, tmp_path):
        d = self._write_config(tmp_path, [{"dockerfile": "Dockerfile"}])
        with pytest.raises(ValueError, match="must have 'name'"):
            load_config(d)

    def test_valid_download_timeout(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cuda",
                    "dockerfile": "Dockerfile.cuda",
                    "download_timeout": 720,
                }
            ],
        )
        result = load_config(d)
        assert result["variants"][0]["download_timeout"] == 720

    def test_invalid_download_timeout_zero(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cpu",
                    "dockerfile": "Dockerfile",
                    "download_timeout": 0,
                }
            ],
        )
        with pytest.raises(ValueError, match="positive integer"):
            load_config(d)

    def test_invalid_download_timeout_negative(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cpu",
                    "dockerfile": "Dockerfile",
                    "download_timeout": -1,
                }
            ],
        )
        with pytest.raises(ValueError, match="positive integer"):
            load_config(d)

    def test_invalid_download_timeout_string(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cpu",
                    "dockerfile": "Dockerfile",
                    "download_timeout": "480",
                }
            ],
        )
        with pytest.raises(ValueError, match="positive integer"):
            load_config(d)


class TestResolvePlatformGroups:
    def test_cli_overrides_config(self):
        groups, source = resolve_platform_groups(["aarch64"], {"platforms": ["x86_64"]})
        assert groups == [PLATFORM_TAGS["aarch64"]]
        assert source == "CLI"

    def test_config_used_when_no_cli(self):
        groups, source = resolve_platform_groups(
            [], {"platforms": ["x86_64", "aarch64"]}
        )
        assert len(groups) == 2
        assert groups[0] == PLATFORM_TAGS["x86_64"]
        assert groups[1] == PLATFORM_TAGS["aarch64"]
        assert source == "variant config"

    def test_default_when_no_cli_no_config(self):
        groups, source = resolve_platform_groups([], {})
        assert groups == [PLATFORM_TAGS["x86_64"]]
        assert source == "default"

    def test_multiple_cli_platforms(self):
        groups, source = resolve_platform_groups(["x86_64", "aarch64"], {})
        assert len(groups) == 2
        assert source == "CLI"

    def test_source_label_cli(self):
        _, source = resolve_platform_groups(["x86_64"], {})
        assert source == "CLI"

    def test_source_label_variant_config(self):
        _, source = resolve_platform_groups([], {"platforms": ["x86_64"]})
        assert source == "variant config"

    def test_source_label_default(self):
        _, source = resolve_platform_groups([], {})
        assert source == "default"

    def test_cli_with_alias(self):
        groups, source = resolve_platform_groups(["amd64"], {})
        assert groups == [PLATFORM_TAGS["x86_64"]]
        assert source == "CLI"

    def test_invalid_platform_in_cli_raises(self):
        with pytest.raises(ValueError, match="Unsupported architecture"):
            resolve_platform_groups(["mips64"], {})


class TestTimeoutResolution:
    def test_cli_timeout_overrides_config(self):
        assert resolve_download_timeout(300, {"download_timeout": 720}) == 300

    def test_config_timeout_used_when_no_cli(self):
        assert resolve_download_timeout(None, {"download_timeout": 720}) == 720

    def test_default_timeout_when_neither_set(self):
        assert resolve_download_timeout(None, {}) == 480


class TestGenerateForIndexRaceCondition:
    def test_failing_group_does_not_delete_temp_dir_early(self, monkeypatch, tmp_path):
        """One group fails fast; verify temp dir still exists for the other."""
        generate_for_index_fn = _mod.generate_for_index
        captured_tmpdir = []
        original_td = _mod.tempfile.TemporaryDirectory

        class _CaptureTmpDir:
            def __init__(self, *args, **kwargs):
                self._inner = original_td(*args, **kwargs)

            def __enter__(self):
                path = self._inner.__enter__()
                captured_tmpdir.append(path)
                return path

            def __exit__(self, *args):
                return self._inner.__exit__(*args)

        call_count = {"slow": 0}

        def mock_run_pip_command(cmd, timeout, phase_name, **kwargs):
            if "Phase 1" in phase_name:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="", stderr=""
                )
            context = kwargs.get("context", "")
            if "ppc64le" in context:
                raise RuntimeError("No ppc64le wheels")
            time.sleep(0.1)
            call_count["slow"] += 1
            if captured_tmpdir:
                assert Path(
                    captured_tmpdir[0]
                ).exists(), "Temp dir was deleted while other groups still running"
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        monkeypatch.setattr(_mod.tempfile, "TemporaryDirectory", _CaptureTmpDir)
        monkeypatch.setattr(_mod, "run_pip_command", mock_run_pip_command)
        monkeypatch.setattr(
            _mod,
            "parse_report_packages_and_hashes",
            lambda **kwargs: ([], {}),
        )
        monkeypatch.setattr(_mod, "collect_hashes_from_download_dir", lambda *a: {})

        platform_groups = [
            PLATFORM_TAGS["x86_64"],
            PLATFORM_TAGS["ppc64le"],
        ]
        result = generate_for_index_fn(
            index_url=None,
            root_names=["dummy"],
            platform_groups=platform_groups,
            out_path=tmp_path / "test-out.txt",
            dry_run=False,
            download_timeout=60,
        )
        assert result == 1

    def test_all_failures_reported(self, monkeypatch, capsys, tmp_path):
        generate_for_index_fn = _mod.generate_for_index

        def mock_run_pip_command(cmd, timeout, phase_name, **kwargs):
            if "Phase 1" in phase_name:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="", stderr=""
                )
            context = kwargs.get("context", "")
            raise RuntimeError(f"Failed for {context}")

        monkeypatch.setattr(_mod, "run_pip_command", mock_run_pip_command)
        monkeypatch.setattr(
            _mod,
            "parse_report_packages_and_hashes",
            lambda **kwargs: ([], {}),
        )

        result = generate_for_index_fn(
            index_url=None,
            root_names=["dummy"],
            platform_groups=[PLATFORM_TAGS["x86_64"], PLATFORM_TAGS["aarch64"]],
            out_path=tmp_path / "test-out.txt",
            dry_run=False,
            download_timeout=60,
        )
        assert result == 1
        captured = capsys.readouterr()
        assert "x86_64" in captured.err or "manylinux2014_x86_64" in captured.err
        assert "aarch64" in captured.err or "manylinux2014_aarch64" in captured.err

    def test_successful_groups_still_collected_on_partial_failure(
        self, monkeypatch, tmp_path
    ):
        generate_for_index_fn = _mod.generate_for_index
        collected_dirs = []

        def mock_run_pip_command(cmd, timeout, phase_name, **kwargs):
            if "Phase 1" in phase_name:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="", stderr=""
                )
            context = kwargs.get("context", "")
            if "ppc64le" in context:
                raise RuntimeError("No ppc64le wheels")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        def mock_collect_hashes(group_dir, hash_cache):
            collected_dirs.append(str(group_dir))
            return {}

        monkeypatch.setattr(_mod, "run_pip_command", mock_run_pip_command)
        monkeypatch.setattr(
            _mod,
            "parse_report_packages_and_hashes",
            lambda **kwargs: ([], {}),
        )
        monkeypatch.setattr(
            _mod, "collect_hashes_from_download_dir", mock_collect_hashes
        )

        result = generate_for_index_fn(
            index_url=None,
            root_names=["dummy"],
            platform_groups=[
                PLATFORM_TAGS["x86_64"],
                PLATFORM_TAGS["ppc64le"],
                PLATFORM_TAGS["aarch64"],
            ],
            out_path=tmp_path / "test-out.txt",
            dry_run=False,
            download_timeout=60,
        )
        assert result == 1
        assert len(collected_dirs) == 2

    def test_proc_registry_cleaned_on_attempt_completion(self):
        registry_list = []
        registry_lock = threading.Lock()
        registry = (registry_list, registry_lock)

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdout = iter(["line1\n"])
            mock_proc.poll.return_value = 0
            mock_proc.returncode = 0
            mock_proc.wait.return_value = 0
            mock_proc.__enter__ = MagicMock(return_value=mock_proc)
            mock_proc.__exit__ = MagicMock(return_value=False)
            mock_popen.return_value = mock_proc

            run_pip_command(
                ["echo", "test"],
                timeout=10,
                phase_name="Test",
                proc_registry=registry,
            )
        assert len(registry_list) == 0

    def test_proc_registry_no_stale_entries_across_retries(self):
        """Registry never retains a proc from a failed attempt during retries."""
        registry_list = []
        registry_lock = threading.Lock()
        registry = (registry_list, registry_lock)

        class _TrackingPopen:
            """Mock Popen that tracks registry size on each creation."""

            def __init__(self, *args, **kwargs):
                self._args = args
                self._kwargs = kwargs
                self.returncode = None
                self.stdout = iter([])
                self._call_count = 0

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def poll(self):
                if self._call_count == 0:
                    self._call_count += 1
                    self.returncode = 1
                    return 1
                self.returncode = 0
                return 0

            def wait(self, timeout=None):
                return self.returncode

        attempt_num = [0]

        with patch("subprocess.Popen") as mock_popen:

            def make_proc(*args, **kwargs):
                attempt_num[0] += 1
                p = MagicMock()
                p.stdout = iter([])
                p.__enter__ = MagicMock(return_value=p)
                p.__exit__ = MagicMock(return_value=False)
                if attempt_num[0] == 1:
                    p.poll.return_value = 1
                    p.returncode = 1
                    p.wait.return_value = 1
                else:
                    p.poll.return_value = 0
                    p.returncode = 0
                    p.wait.return_value = 0
                return p

            mock_popen.side_effect = make_proc

            run_pip_command(
                ["echo", "test"],
                timeout=10,
                phase_name="Test",
                attempts=2,
                retry_backoff_sec=0,
                proc_registry=registry,
            )

        assert len(registry_list) == 0

    def test_sigterm_sent_before_sigkill(self):
        mock_proc_fast = MagicMock()
        mock_proc_fast.terminate.return_value = None
        mock_proc_fast.wait.return_value = 0

        mock_proc_slow = MagicMock()
        mock_proc_slow.terminate.return_value = None
        mock_proc_slow.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="x", timeout=5),
            0,
        ]
        mock_proc_slow.kill.return_value = None

        procs = [mock_proc_fast, mock_proc_slow]
        lock = threading.Lock()

        _kill_registered_procs(procs, lock)

        mock_proc_fast.terminate.assert_called_once()
        mock_proc_fast.wait.assert_called_once_with(timeout=5)

        mock_proc_slow.terminate.assert_called_once()
        mock_proc_slow.kill.assert_called_once()
        assert mock_proc_slow.wait.call_count == 2


class TestDryRun:
    def test_dry_run_uses_resolved_platform_tags(self, capsys, monkeypatch, tmp_path):
        generate_for_index_fn = _mod.generate_for_index
        monkeypatch.setattr(_mod, "_pip_supports_report", lambda: (True, ""))
        result = generate_for_index_fn(
            index_url="https://example.com/simple",
            root_names=["mlserver"],
            platform_groups=[PLATFORM_TAGS["x86_64"]],
            out_path=tmp_path / "dry-out.txt",
            dry_run=True,
            download_timeout=480,
        )
        assert result == 0
        captured = capsys.readouterr()
        assert "--platform manylinux2014_x86_64" in captured.out
        assert "--platform manylinux_2_34_x86_64" in captured.out
        assert "--platform linux_x86_64" in captured.out

    def test_dry_run_respects_variant_platforms(self, capsys, monkeypatch, tmp_path):
        generate_for_index_fn = _mod.generate_for_index
        monkeypatch.setattr(_mod, "_pip_supports_report", lambda: (True, ""))
        result = generate_for_index_fn(
            index_url="https://example.com/simple",
            root_names=["mlserver"],
            platform_groups=[PLATFORM_TAGS["aarch64"]],
            out_path=tmp_path / "dry-out.txt",
            dry_run=True,
            download_timeout=480,
        )
        assert result == 0
        captured = capsys.readouterr()
        assert "manylinux2014_aarch64" in captured.out
        assert "manylinux2014_x86_64" not in captured.out

    def test_dry_run_returns_zero(self, capsys, monkeypatch, tmp_path):
        generate_for_index_fn = _mod.generate_for_index
        monkeypatch.setattr(_mod, "_pip_supports_report", lambda: (True, ""))
        result = generate_for_index_fn(
            index_url=None,
            root_names=["mlserver"],
            platform_groups=[PLATFORM_TAGS["x86_64"], PLATFORM_TAGS["s390x"]],
            out_path=tmp_path / "dry-out.txt",
            dry_run=True,
            download_timeout=480,
        )
        assert result == 0


class TestHelpers:
    def test_normalize_distribution_name(self):
        assert normalize_distribution_name("My_Package") == "my-package"
        assert normalize_distribution_name("some.pkg") == "some-pkg"
        assert normalize_distribution_name("foo") == "foo"

    def test_redact_index_url_with_password(self):
        url = "https://user:secret@pypi.example.com/simple?token=abc123"
        result = redact_index_url(url)
        assert "secret" not in result
        assert "abc123" not in result
        assert "pypi.example.com" in result

    def test_redact_index_url_without_password(self):
        url = "https://pypi.example.com/simple"
        result = redact_index_url(url)
        assert result == url

    def test_name_version_from_filename_wheel(self):
        p = Path("foo-1.2.3-py3-none-any.whl")
        result = _name_version_from_filename(p)
        assert result == ("foo", "1.2.3")

    def test_name_version_from_filename_sdist(self):
        p = Path("foo-1.2.3.tar.gz")
        result = _name_version_from_filename(p)
        assert result == ("foo", "1.2.3")

    def test_name_version_from_filename_unparseable(self):
        p = Path("not-a-valid-format.txt")
        result = _name_version_from_filename(p)
        assert result is None

    def test_get_base_image_from_dockerfile(self, tmp_path):
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text(
            'ARG BASE_IMAGE="registry.example.com/image:latest"\n'
            "FROM ${BASE_IMAGE}\n"
            "RUN echo hello\n"
        )
        result = get_base_image_from_dockerfile(tmp_path, "Dockerfile")
        assert result == "registry.example.com/image:latest"

    def test_get_base_image_from_dockerfile_no_from(self, tmp_path):
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("ARG FOO=bar\n")
        with pytest.raises(ValueError, match="FROM"):
            get_base_image_from_dockerfile(tmp_path, "Dockerfile")


class TestBuildRequirementStrings:
    def test_basic_object_format(self):
        pip_reqs, bare = build_requirement_strings(
            [{"name": "mlserver", "version": "1.7.0"}]
        )
        assert pip_reqs == ["mlserver==1.7.0"]
        assert bare == ["mlserver"]

    def test_object_with_extras(self):
        pip_reqs, bare = build_requirement_strings(
            [{"name": "mlserver-onnx", "version": "1.7.0", "extras": ["cpu"]}]
        )
        assert pip_reqs == ["mlserver-onnx[cpu]==1.7.0"]
        assert bare == ["mlserver-onnx[cpu]"]

    def test_object_with_multiple_extras(self):
        pip_reqs, bare = build_requirement_strings(
            [
                {
                    "name": "mlserver-onnx",
                    "version": "1.7.0",
                    "extras": ["cpu", "rocm"],
                }
            ]
        )
        assert pip_reqs == ["mlserver-onnx[cpu,rocm]==1.7.0"]
        assert bare == ["mlserver-onnx[cpu,rocm]"]

    def test_legacy_string_format(self):
        pip_reqs, bare = build_requirement_strings(["mlserver", "mlserver-onnx[cpu]"])
        assert pip_reqs == ["mlserver", "mlserver-onnx[cpu]"]
        assert bare == ["mlserver", "mlserver-onnx[cpu]"]

    def test_mixed_formats(self):
        pip_reqs, bare = build_requirement_strings(
            [
                "mlserver-legacy",
                {"name": "mlserver", "version": "1.7.0"},
            ]
        )
        assert pip_reqs == ["mlserver-legacy", "mlserver==1.7.0"]
        assert bare == ["mlserver-legacy", "mlserver"]

    def test_returns_tuple_of_two_lists(self):
        result = build_requirement_strings([{"name": "mlserver", "version": "1.0"}])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_bare_names_omit_version(self):
        _, bare = build_requirement_strings(
            [
                {"name": "mlserver", "version": "1.7.1+rhaiv.11"},
                {
                    "name": "mlserver-onnx",
                    "version": "1.7.1+rhaiv.11",
                    "extras": ["cpu"],
                },
            ]
        )
        for name in bare:
            assert "==" not in name
            assert "1.7.1" not in name

    def test_empty_list(self):
        pip_reqs, bare = build_requirement_strings([])
        assert pip_reqs == []
        assert bare == []

    def test_object_without_extras(self):
        pip_reqs, bare = build_requirement_strings(
            [{"name": "mlserver", "version": "2.0", "extras": []}]
        )
        assert pip_reqs == ["mlserver==2.0"]
        assert bare == ["mlserver"]


class TestLoadConfigRootPackagesValidation:
    def _write_config(self, tmp_path, variants):
        config = {"variants": variants}
        config_path = tmp_path / "requirements-config.json"
        config_path.write_text(json.dumps(config))
        return tmp_path

    def test_valid_object_format(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cpu",
                    "dockerfile": "Dockerfile",
                    "root_packages": [
                        {"name": "mlserver", "version": "1.7.0"},
                        {
                            "name": "mlserver-onnx",
                            "version": "1.7.0",
                            "extras": ["cpu"],
                        },
                    ],
                }
            ],
        )
        result = load_config(d)
        rp = result["variants"][0]["root_packages"]
        assert len(rp) == 2
        assert rp[0]["name"] == "mlserver"
        assert rp[1]["extras"] == ["cpu"]

    def test_legacy_string_format_accepted(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cpu",
                    "dockerfile": "Dockerfile",
                    "root_packages": ["mlserver", "mlserver-onnx[cpu]"],
                }
            ],
        )
        result = load_config(d)
        assert result["variants"][0]["root_packages"] == [
            "mlserver",
            "mlserver-onnx[cpu]",
        ]

    def test_missing_name_raises(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cpu",
                    "dockerfile": "Dockerfile",
                    "root_packages": [{"version": "1.0"}],
                }
            ],
        )
        with pytest.raises(ValueError, match="string 'name'"):
            load_config(d)

    def test_missing_version_raises(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cpu",
                    "dockerfile": "Dockerfile",
                    "root_packages": [{"name": "mlserver"}],
                }
            ],
        )
        with pytest.raises(ValueError, match="string 'version'"):
            load_config(d)

    def test_invalid_extras_type_raises(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cpu",
                    "dockerfile": "Dockerfile",
                    "root_packages": [
                        {"name": "mlserver", "version": "1.0", "extras": "cpu"}
                    ],
                }
            ],
        )
        with pytest.raises(ValueError, match="'extras' must be a list"):
            load_config(d)

    def test_empty_extras_string_raises(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cpu",
                    "dockerfile": "Dockerfile",
                    "root_packages": [
                        {"name": "mlserver", "version": "1.0", "extras": [""]}
                    ],
                }
            ],
        )
        with pytest.raises(ValueError, match="'extras' must be a list"):
            load_config(d)

    def test_invalid_entry_type_raises(self, tmp_path):
        d = self._write_config(
            tmp_path,
            [
                {
                    "name": "cpu",
                    "dockerfile": "Dockerfile",
                    "root_packages": [123],
                }
            ],
        )
        with pytest.raises(ValueError, match="string or object"):
            load_config(d)


class TestSkipGracefully:
    def test_skip_when_root_package_not_found(self, monkeypatch, tmp_path):
        generate_for_index_fn = _mod.generate_for_index

        def mock_run_pip_command(cmd, timeout, phase_name, **kwargs):
            raise RuntimeError(
                "Phase 1 failed with exit code 1. "
                "Suspected requirement: mlserver==1.7.1+rhaiv.12.\n"
                "stdout(last): No matching distribution found for "
                "mlserver==1.7.1+rhaiv.12\n"
                "stderr(last): <merged-into-stdout>"
            )

        monkeypatch.setattr(_mod, "run_pip_command", mock_run_pip_command)
        monkeypatch.setattr(_mod, "_pip_supports_report", lambda: (True, ""))

        result = generate_for_index_fn(
            index_url=None,
            root_names=["mlserver==1.7.1+rhaiv.12"],
            root_bare_names=["mlserver"],
            platform_groups=[PLATFORM_TAGS["x86_64"]],
            out_path=tmp_path / "test-out.txt",
            dry_run=False,
            download_timeout=60,
        )
        assert result == SKIP_NOT_AVAILABLE

    def test_no_skip_on_resolution_impossible(self, monkeypatch, tmp_path):
        generate_for_index_fn = _mod.generate_for_index

        def mock_run_pip_command(cmd, timeout, phase_name, **kwargs):
            raise RuntimeError(
                "Phase 1 failed with exit code 1.\n"
                "stdout(last): ResolutionImpossible: conflicting deps "
                "for requirements mlserver==1.7.1\n"
                "stderr(last): <merged-into-stdout>"
            )

        monkeypatch.setattr(_mod, "run_pip_command", mock_run_pip_command)
        monkeypatch.setattr(_mod, "_pip_supports_report", lambda: (True, ""))

        result = generate_for_index_fn(
            index_url=None,
            root_names=["mlserver==1.7.1"],
            root_bare_names=["mlserver"],
            platform_groups=[PLATFORM_TAGS["x86_64"]],
            out_path=tmp_path / "test-out.txt",
            dry_run=False,
            download_timeout=60,
        )
        assert result == 1

    def test_no_skip_when_transitive_dep_not_found(self, monkeypatch, tmp_path):
        generate_for_index_fn = _mod.generate_for_index

        def mock_run_pip_command(cmd, timeout, phase_name, **kwargs):
            raise RuntimeError(
                "Phase 1 failed with exit code 1. "
                "Suspected requirement: numpy>=2.0.\n"
                "stdout(last): No matching distribution found for "
                "numpy>=2.0\n"
                "stderr(last): <merged-into-stdout>"
            )

        monkeypatch.setattr(_mod, "run_pip_command", mock_run_pip_command)
        monkeypatch.setattr(_mod, "_pip_supports_report", lambda: (True, ""))

        result = generate_for_index_fn(
            index_url=None,
            root_names=["mlserver==1.7.1"],
            root_bare_names=["mlserver"],
            platform_groups=[PLATFORM_TAGS["x86_64"]],
            out_path=tmp_path / "test-out.txt",
            dry_run=False,
            download_timeout=60,
        )
        assert result == 1

    def test_skip_with_extras_in_root_package(self, monkeypatch, tmp_path):
        generate_for_index_fn = _mod.generate_for_index

        def mock_run_pip_command(cmd, timeout, phase_name, **kwargs):
            raise RuntimeError(
                "Phase 1 failed with exit code 1.\n"
                "stdout(last): No matching distribution found for "
                "mlserver-onnx==1.7.1+rhaiv.12\n"
                "stderr(last): <merged-into-stdout>"
            )

        monkeypatch.setattr(_mod, "run_pip_command", mock_run_pip_command)
        monkeypatch.setattr(_mod, "_pip_supports_report", lambda: (True, ""))

        result = generate_for_index_fn(
            index_url=None,
            root_names=["mlserver-onnx[cpu]==1.7.1+rhaiv.12"],
            root_bare_names=["mlserver-onnx[cpu]"],
            platform_groups=[PLATFORM_TAGS["x86_64"]],
            out_path=tmp_path / "test-out.txt",
            dry_run=False,
            download_timeout=60,
        )
        assert result == SKIP_NOT_AVAILABLE
