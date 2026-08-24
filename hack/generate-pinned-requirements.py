#!/usr/bin/env python3
"""
Generate requirements-<variant>.txt with pinned versions and SHA256 hashes
for target architectures.

Run inside the base image container for each variant defined in
hack/requirements-config.json. Each variant specifies its own root_packages,
dockerfile, target platforms, and download timeout. The image's pip index is
used for resolution.

Platform configuration:
  Each variant may specify a "platforms" list in requirements-config.json.
  Accepted values: x86_64, aarch64, ppc64le, s390x (canonical names) or
  aliases amd64 (-> x86_64), arm64 (-> aarch64).
  If omitted, defaults to x86_64 only.
  Resolution precedence: --platform CLI > config "platforms" > default.

Options:
  --variant NAME       (required) Variant from config (e.g. cpu, cuda).
  -o PATH              Output path (default: requirements.txt).
  --platform ARCH      Target architecture (can repeat). Overrides config.
  --timeout SECONDS    Download timeout per platform group. Overrides config.
  --index-url URL      Override package index URL.
  --print-base-image   Print base image from Dockerfile.
  --dry-run            Print pip commands without executing.

Usage (in container):
  python hack/generate-pinned-requirements.py \\
    --variant cpu -o requirements/requirements-cpu.txt
  python hack/generate-pinned-requirements.py \\
    --variant cuda -o requirements/requirements-cuda.txt
Usage (on host):
  python hack/generate-pinned-requirements.py --print-base-image Dockerfile.konflux
  python hack/generate-pinned-requirements.py --print-base-image Dockerfile.konflux.cuda
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urlparse


PLATFORM_TAGS: dict[str, list[str]] = {
    "x86_64": ["manylinux2014_x86_64", "manylinux_2_34_x86_64", "linux_x86_64"],
    "aarch64": ["manylinux2014_aarch64", "manylinux_2_34_aarch64", "linux_aarch64"],
    "ppc64le": ["manylinux2014_ppc64le", "manylinux_2_34_ppc64le", "linux_ppc64le"],
    "s390x": ["manylinux2014_s390x", "manylinux_2_34_s390x", "linux_s390x"],
}

PLATFORM_ALIASES: dict[str, str] = {
    "amd64": "x86_64",
    "arm64": "aarch64",
}

DEFAULT_PLATFORM = "x86_64"


def resolve_platform(name: str) -> list[str]:
    """Resolve a short architecture name to its pip platform tag list.

    Args:
        name: Architecture name (e.g. ``x86_64``, ``amd64``, ``aarch64``).

    Returns:
        List of pip platform tags for the architecture.

    Raises:
        ValueError: If the name is not a supported architecture or alias.
    """
    canonical = PLATFORM_ALIASES.get(name, name)
    if canonical in PLATFORM_TAGS:
        return PLATFORM_TAGS[canonical]
    supported = sorted(set(list(PLATFORM_TAGS.keys()) + list(PLATFORM_ALIASES.keys())))
    raise ValueError(
        f"Unsupported architecture '{name}'. " f"Supported: {', '.join(supported)}."
    )


def resolve_platform_groups(
    cli_platforms: list[str], variant: dict
) -> tuple[list[list[str]], str]:
    """Resolve platform groups from CLI, config, or default.

    Returns:
        Tuple of (platform_groups, source_label) where source_label
        is "CLI", "variant config", or "default" for logging.
    """
    if cli_platforms:
        return [resolve_platform(p) for p in cli_platforms], "CLI"
    if "platforms" in variant:
        return [resolve_platform(p) for p in variant["platforms"]], "variant config"
    return [resolve_platform(DEFAULT_PLATFORM)], "default"


def resolve_download_timeout(cli_timeout: int | None, variant: dict) -> int:
    """Resolve download timeout: CLI > config download_timeout > 480."""
    if cli_timeout is not None:
        return cli_timeout
    return variant.get("download_timeout", 480)


def _kill_registered_procs(procs: list[subprocess.Popen], lock: threading.Lock) -> None:
    """Gracefully terminate all registered processes (SIGTERM -> wait -> SIGKILL)."""
    with lock:
        targets = list(procs)
    for p in targets:
        try:
            p.terminate()
        except OSError:
            pass
    for p in targets:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
                p.wait(timeout=5)
            except OSError:
                pass


CONFIG_FILENAME = "requirements-config.json"
SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "password",
    "token",
}


def normalize_distribution_name(name: str) -> str:
    """Return the PEP 503 normalized form of a distribution name.

    Args:
        name: Raw package/distribution name.

    Returns:
        The canonical name, lowercased with ``_`` and ``.`` replaced by ``-``.
    """
    return name.lower().replace("_", "-").replace(".", "-")


SKIP_NOT_AVAILABLE = 2
"""Return code from :func:`generate_for_index` when a root package version
is not yet published to the index.  The caller should write a skip marker
file and exit successfully so CI stays green."""


def build_requirement_strings(
    root_packages: list[dict | str],
) -> tuple[list[str], list[str]]:
    """Convert config ``root_packages`` to pip requirement strings.

    Handles both the legacy string format (``"mlserver"``) and the new
    object format
    (``{"name": "mlserver", "version": "1.7.0", "extras": ["cpu"]}``).

    Args:
        root_packages: List of root package entries from config.

    Returns:
        Tuple of ``(pip_requirement_strings, bare_name_strings)``.
        Pip strings include version pins (e.g. ``mlserver[cpu]==1.7.0``).
        Bare strings omit the version (e.g. ``mlserver[cpu]``).
    """
    pip_reqs: list[str] = []
    bare_names: list[str] = []
    for pkg in root_packages:
        if isinstance(pkg, str):
            pip_reqs.append(pkg)
            bare_names.append(pkg)
        elif isinstance(pkg, dict):
            name = pkg["name"]
            version = pkg["version"]
            extras = pkg.get("extras", [])
            extras_str = f"[{','.join(extras)}]" if extras else ""
            bare = f"{name}{extras_str}"
            pip_reqs.append(f"{bare}=={version}")
            bare_names.append(bare)
    return pip_reqs, bare_names


def redact_index_url(url: str) -> str:
    """Redact secrets from an index URL before logging or writing output.

    This function removes userinfo (``username:password@``) from the netloc and
    masks values for sensitive query parameters such as ``token`` and
    ``password``.

    Args:
        url: Raw index URL that may contain credentials or tokens.

    Returns:
        A sanitized URL safe for logs and generated requirement headers.
    """
    parsed = urlparse(url)
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"

    redacted_qs = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in SENSITIVE_QUERY_KEYS:
            redacted_qs.append((key, "***"))
        else:
            redacted_qs.append((key, value))

    return parsed._replace(netloc=netloc, query=urlencode(redacted_qs)).geturl()


def sanitize_log_line(line: str) -> str:
    """Redact URL-like secrets from a pip log line.

    Args:
        line: A single log line emitted by pip.

    Returns:
        A sanitized log line with URL credentials and sensitive query values
        masked.
    """
    # pip commonly prints: "Looking in indexes: <url>, <url>"
    if "Looking in indexes:" in line:
        prefix, value = line.split("Looking in indexes:", 1)
        parts = [p.strip() for p in value.split(",")]
        redacted = ", ".join(
            redact_index_url(p) if "://" in p else p for p in parts if p
        )
        return f"{prefix}Looking in indexes: {redacted}"
    # Fallback: redact any URL-looking token inline.
    return re.sub(r"https?://\S+", lambda m: redact_index_url(m.group(0)), line)


def format_command_for_log(cmd: list[object]) -> str:
    """Format a command for display with sensitive index URLs redacted.

    Args:
        cmd: Command tokens that may include ``--index-url``.

    Returns:
        A shell-like command string suitable for safe logging.
    """
    rendered: list[str] = []
    i = 0
    while i < len(cmd):
        token = str(cmd[i])
        if token == "--index-url" and i + 1 < len(cmd):
            rendered.append(token)
            rendered.append(redact_index_url(str(cmd[i + 1])))
            i += 2
            continue
        if token.startswith("--index-url="):
            key, raw = token.split("=", 1)
            rendered.append(f"{key}={redact_index_url(raw)}")
            i += 1
            continue
        rendered.append(token)
        i += 1
    return " ".join(rendered)


def get_system_index_url() -> str | None:
    """Resolve index URL from environment variables or pip configuration.

    Returns:
        The configured index URL when discoverable; otherwise ``None``.
    """
    url = (
        os.environ.get("PIP_INDEX_URL", "").strip()
        or os.environ.get("PIP_EXTRA_INDEX_URL", "").strip()
    )
    if url:
        return url
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "config", "get", "global.index-url"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout and result.stdout.strip():
            return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(
            "  Warning: timed out reading pip global.index-url from pip config.",
            file=sys.stderr,
        )
    except FileNotFoundError:
        print(
            "  Warning: pip executable not found while reading pip config.",
            file=sys.stderr,
        )
    return None


def load_config(script_dir: Path) -> dict:
    """Load and validate requirements generation configuration.

    Args:
        script_dir: Directory containing ``requirements-config.json``.

    Returns:
        Parsed configuration dictionary with validated ``variants`` key.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If required keys or values are missing/invalid.
    """
    config_path = script_dir / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    data = json.loads(config_path.read_text())
    if "variants" not in data:
        raise ValueError("Config missing required key: 'variants'")
    variants = data["variants"]
    if not isinstance(variants, list) or not variants:
        raise ValueError("Config 'variants' must be a non-empty list")
    for i, ent in enumerate(variants):
        if not isinstance(ent, dict) or "name" not in ent:
            raise ValueError(f"Config 'variants'][{i}] must have 'name'")
        if "dockerfile" not in ent or not ent.get("dockerfile"):
            raise ValueError(
                f"Config 'variants'][{i}] must have non-empty 'dockerfile' "
                "(path from repo root)"
            )
        if "platforms" in ent:
            platforms = ent["platforms"]
            if not isinstance(platforms, list) or not platforms:
                raise ValueError(
                    f"Config 'variants'][{i}] 'platforms' must be a non-empty list"
                )
            for p in platforms:
                if not isinstance(p, str):
                    raise ValueError(
                        f"Config 'variants'][{i}] 'platforms' entries must be strings"
                    )
                resolve_platform(p)
            if "cuda" in ent.get("name", "").lower() and "ppc64le" in platforms:
                print(
                    f"  Warning: variant '{ent['name']}' includes ppc64le but "
                    "CUDA indexes typically lack ppc64le wheels.",
                    file=sys.stderr,
                )
        if "root_packages" in ent:
            rp = ent["root_packages"]
            if not isinstance(rp, list):
                raise ValueError(
                    f"Config 'variants'][{i}] 'root_packages' must be a list"
                )
            for j, pkg in enumerate(rp):
                if isinstance(pkg, str):
                    if not pkg.strip():
                        raise ValueError(
                            f"Config 'variants'][{i}] 'root_packages'[{j}] "
                            "must be a non-empty string"
                        )
                    continue
                if not isinstance(pkg, dict):
                    raise ValueError(
                        f"Config 'variants'][{i}] 'root_packages'[{j}] "
                        "must be a string or object"
                    )
                if "name" not in pkg or not isinstance(pkg["name"], str):
                    raise ValueError(
                        f"Config 'variants'][{i}] 'root_packages'[{j}] "
                        "must have a string 'name'"
                    )
                if not pkg["name"].strip():
                    raise ValueError(
                        f"Config 'variants'][{i}] 'root_packages'[{j}] "
                        "'name' must not be blank"
                    )
                if "version" not in pkg or not isinstance(pkg["version"], str):
                    raise ValueError(
                        f"Config 'variants'][{i}] 'root_packages'[{j}] "
                        "must have a string 'version'"
                    )
                if not pkg["version"].strip():
                    raise ValueError(
                        f"Config 'variants'][{i}] 'root_packages'[{j}] "
                        "'version' must not be blank"
                    )
                if "extras" in pkg:
                    extras = pkg["extras"]
                    if not isinstance(extras, list) or not all(
                        isinstance(e, str) and e.strip() for e in extras
                    ):
                        raise ValueError(
                            f"Config 'variants'][{i}] "
                            f"'root_packages'[{j}] 'extras' must be a "
                            "list of non-empty strings"
                        )
        if "download_timeout" in ent:
            dt = ent["download_timeout"]
            if not isinstance(dt, int) or dt <= 0:
                raise ValueError(
                    f"Config 'variants'][{i}] 'download_timeout' must be a "
                    "positive integer (seconds)"
                )
    return data


_ARG_RE = re.compile(
    r"^\s*ARG\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s*=\s*(\"[^\"]*\"|'[^']*'|[^ \t#]+))?\s*$",
    re.IGNORECASE,
)
_FROM_RE = re.compile(
    r"^\s*FROM(?:\s+--platform=\S+)?\s+(\S+)(?:\s+AS\s+\S+)?\s*$",
    re.IGNORECASE,
)


def _strip_unquoted_comment(line: str) -> str:
    """Remove trailing comment content from a Dockerfile line.

    Args:
        line: Raw Dockerfile line.

    Returns:
        The trimmed line content before ``#``.
    """
    return line.split("#", 1)[0].strip()


def get_base_image_from_dockerfile(repo_root: Path, dockerfile_path: str) -> str:
    """Extract and resolve the first ``FROM`` base image from a Dockerfile.

    ``ARG`` default values defined before ``FROM`` are applied to ``${VAR}``
    placeholders in the image reference.

    Args:
        repo_root: Repository root used as the Dockerfile path base.
        dockerfile_path: Path to the Dockerfile relative to ``repo_root``.

    Returns:
        The resolved base image reference from the first ``FROM`` instruction.

    Raises:
        FileNotFoundError: If the Dockerfile path does not exist.
        ValueError: If no ``FROM`` is found or substitutions remain unresolved.
    """
    df_path = (repo_root / dockerfile_path).resolve()
    if not df_path.is_relative_to(repo_root.resolve()):
        raise ValueError(f"Dockerfile path escapes repo root: {dockerfile_path}")
    if not df_path.exists():
        raise FileNotFoundError(f"Dockerfile not found: {df_path}")
    text = df_path.read_text()
    args: dict[str, str] = {}
    from_line: str | None = None
    for line in text.splitlines():
        clean = _strip_unquoted_comment(line)
        if not clean:
            continue
        arg_m = _ARG_RE.match(clean)
        if arg_m:
            name, raw_value = arg_m.groups()
            value = (raw_value or "").strip()
            if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                value = value[1:-1]
            if value:
                args[name] = value
            continue
        from_m = _FROM_RE.match(clean)
        if from_m:
            from_line = from_m.group(1)
            break
    if not from_line:
        raise ValueError(f"No FROM found in {df_path}")

    # Substitute ${VAR} with args
    def repl(m: re.Match) -> str:
        """Resolve one ``${VAR}`` token using collected ARG defaults."""
        var = m.group(1)
        return args.get(var, m.group(0))

    resolved = re.sub(r"\$\{(\w+)\}", repl, from_line)
    if "${" in resolved:
        raise ValueError(f"Unresolved variable(s) in FROM in {df_path}: {from_line}")
    return resolved.strip()


def sha256_file(path: Path) -> str:
    """Compute SHA256 for a file.

    Args:
        path: File path to hash.

    Returns:
        Hex-encoded SHA256 digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_report(report_path: Path) -> dict:
    """Load a pip JSON report and validate top-level type.

    Args:
        report_path: Path to the JSON report file.

    Returns:
        Parsed report as a dictionary.

    Raises:
        ValueError: If the parsed JSON root is not a dictionary.
    """
    data = json.loads(report_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid pip report format in {report_path}")
    return data


def _report_filename_for_item(item: dict) -> str | None:
    """Return artifact filename from a pip report ``install`` item.

    Args:
        item: Single entry from the report ``install`` list.

    Returns:
        Artifact filename if available; otherwise ``None``.
    """
    download_info = item.get("download_info") or {}
    if not isinstance(download_info, dict):
        return None
    url = download_info.get("url")
    if not isinstance(url, str) or not url:
        return None
    return Path(unquote(urlparse(url).path)).name or None


def _name_version_from_filename(path: Path) -> tuple[str, str] | None:
    """Parse normalized package name and version from artifact filename.

    Args:
        path: Downloaded wheel/sdist artifact path.

    Returns:
        ``(normalized_name, version)`` when parsable; otherwise ``None``.
    """
    name = path.name
    if name.endswith(".whl"):
        parts = name[:-4].split("-")
        if len(parts) >= 2:
            return normalize_distribution_name(parts[0]), parts[1]
        return None
    for suffix in (".tar.gz", ".tar.bz2", ".zip", ".tar.xz"):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            for i in range(len(stem) - 1, -1, -1):
                if stem[i] == "-" and i + 1 < len(stem) and stem[i + 1].isdigit():
                    pkg = normalize_distribution_name(stem[:i])
                    ver = stem[i + 1 :]
                    if pkg and ver:
                        return pkg, ver
            return None
    return None


def collect_hashes_from_download_dir(
    download_dir: Path, hash_cache: dict[str, str]
) -> dict[tuple[str, str], set[str]]:
    """Collect package hashes from a directory of downloaded artifacts.

    Args:
        download_dir: Directory containing wheel/sdist artifacts.
        hash_cache: Cache of ``absolute_path -> sha256`` to avoid re-hashing.

    Returns:
        Mapping of ``(normalized_name, version)`` to SHA256 digest set.

    Raises:
        ValueError: If no files exist or filenames cannot be parsed reliably.
    """
    hashes_by_package: dict[tuple[str, str], set[str]] = {}
    seen_files = 0
    unparseable_files: list[str] = []
    for artifact in download_dir.iterdir():
        if not artifact.is_file():
            continue
        seen_files += 1
        key = _name_version_from_filename(artifact)
        if not key:
            unparseable_files.append(artifact.name)
            continue
        cache_key = str(artifact.resolve())
        digest = hash_cache.get(cache_key)
        if not digest:
            digest = sha256_file(artifact)
            hash_cache[cache_key] = digest
        hashes_by_package.setdefault(key, set()).add(digest)
    if seen_files == 0:
        raise ValueError(f"No artifacts found in download dir: {download_dir}")
    if unparseable_files:
        raise ValueError(
            "Could not parse package/version from downloaded artifacts: "
            + ", ".join(sorted(unparseable_files))
        )
    return hashes_by_package


def parse_report_packages_and_hashes(
    report_path: Path,
    download_dir: Path,
    hash_cache: dict[str, str] | None = None,
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], set[str]]]:
    """Parse resolved packages and hashes from a pip ``--report`` output.

    Hashes from report metadata are preferred. When missing, local artifacts in
    ``download_dir`` are hashed as a fallback if present.

    Args:
        report_path: Path to the pip JSON report.
        download_dir: Directory that may contain artifacts referenced by report.
        hash_cache: Optional file-hash cache shared across phases.

    Returns:
        A tuple of:
        - ordered list of ``(normalized_name, version)`` packages
        - mapping of ``(normalized_name, version)`` to hash set

    Raises:
        ValueError: If report structure or required metadata is invalid.
    """
    hash_cache = hash_cache if hash_cache is not None else {}
    report = _load_report(report_path)
    install_items = report.get("install")
    if not isinstance(install_items, list):
        raise ValueError(
            f"Invalid pip report: missing/invalid 'install' list in {report_path}"
        )
    seen: set[tuple[str, str]] = set()
    packages: list[tuple[str, str]] = []
    hashes_by_package: dict[tuple[str, str], set[str]] = {}
    for idx, item in enumerate(install_items):
        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid pip report item at install[{idx}]: expected object"
            )
        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Invalid pip report item at install[{idx}]: missing/invalid metadata"
            )
        name = metadata.get("name")
        version = metadata.get("version")
        if not name or not version:
            raise ValueError(
                f"Invalid pip report item at install[{idx}]: missing name/version"
            )
        key = (normalize_distribution_name(str(name)), str(version))
        if key not in seen:
            seen.add(key)
            packages.append(key)

        download_info = item.get("download_info") or {}
        archive_info = (
            download_info.get("archive_info")
            if isinstance(download_info, dict)
            else None
        )
        hashes = archive_info.get("hashes") if isinstance(archive_info, dict) else None
        sha_from_report = hashes.get("sha256") if isinstance(hashes, dict) else None
        if isinstance(sha_from_report, str) and sha_from_report:
            hashes_by_package.setdefault(key, set()).add(sha_from_report)
            continue

        filename = _report_filename_for_item(item)
        if not filename:
            continue
        local_path = download_dir / filename
        if local_path.exists():
            cache_key = str(local_path.resolve())
            local_hash = hash_cache.get(cache_key)
            if not local_hash:
                local_hash = sha256_file(local_path)
                hash_cache[cache_key] = local_hash
            hashes_by_package.setdefault(key, set()).add(local_hash)
    return packages, hashes_by_package


def _extract_failed_requirement(stderr_text: str) -> str | None:
    """Extract a likely failing requirement from pip error text.

    Args:
        stderr_text: Combined pip error/output text.

    Returns:
        Requirement token when pattern matching succeeds; otherwise ``None``.
    """
    patterns = [
        r"Could not find a version that satisfies the requirement\s+([^\s(]+)",
        r"No matching distribution found for\s+([^\s]+)",
        r"ResolutionImpossible:.*?\n.*?for requirements?\s+([^\s,]+)",
    ]
    for pat in patterns:
        match = re.search(pat, stderr_text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1)
    return None


def _pip_supports_report(min_major: int = 22, min_minor: int = 2) -> tuple[bool, str]:
    """Check whether local pip supports ``--report``.

    Args:
        min_major: Minimum supported major pip version.
        min_minor: Minimum supported minor pip version.

    Returns:
        ``(True, version_output)`` when supported, otherwise
        ``(False, reason)``.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, f"Unable to run pip --version: {e}"
    if result.returncode != 0:
        err = (result.stderr or "").strip() or "<empty>"
        return False, f"pip --version failed with exit code {result.returncode}: {err}"
    output = (result.stdout or "").strip()
    match = re.search(r"\bpip\s+(\d+)\.(\d+)(?:\.(\d+))?\b", output)
    if not match:
        return False, f"Could not parse pip version from: {output or '<empty>'}"
    major = int(match.group(1))
    minor = int(match.group(2))
    if (major, minor) < (min_major, min_minor):
        return (
            False,
            f"Detected pip {major}.{minor}; this script requires "
            f"pip >= {min_major}.{min_minor} because it uses `pip --report`.",
        )
    return True, output


def run_pip_command(
    cmd: list[object],
    timeout: int,
    phase_name: str,
    context: str = "",
    attempts: int = 1,
    retry_backoff_sec: int = 0,
    proc_registry: tuple[list[subprocess.Popen], threading.Lock] | None = None,
) -> subprocess.CompletedProcess:
    """Run a pip command with streamed logs, timeout enforcement, and retries.

    Args:
        cmd: Command tokens to execute.
        timeout: Per-attempt timeout in seconds.
        phase_name: Label used in log/error messages.
        context: Optional context suffix for logs (for example platform group).
        attempts: Number of attempts before failing.
        retry_backoff_sec: Delay between failed attempts in seconds.
        proc_registry: Optional (list, lock) tuple for registering live procs
            so they can be cleaned up on signal interruption.

    Returns:
        ``subprocess.CompletedProcess`` containing merged stdout on success.

    Raises:
        ValueError: If ``attempts < 1``.
        RuntimeError: If all attempts fail or time out.
    """
    cmd_args = [str(x) for x in cmd]
    where = f" ({context})" if context else ""
    phase_tag = f"[{phase_name}]"
    context_tag = f" [{context}]" if context else ""
    prefix = f"{phase_tag}{context_tag}"

    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    last_error: RuntimeError | None = None
    for attempt in range(1, attempts + 1):
        out_lines: list[str] = []
        proc = None
        attempt_prefix = (
            f"{prefix} [attempt {attempt}/{attempts}]" if attempts > 1 else prefix
        )
        try:
            with subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
            ) as proc:
                if proc_registry is not None:
                    procs_list, procs_lock = proc_registry
                    with procs_lock:
                        procs_list.append(proc)
                assert proc.stdout is not None
                line_queue: queue.Queue[str | None] = queue.Queue()
                stdout_stream = proc.stdout

                def _reader(stream=stdout_stream, out_queue=line_queue) -> None:
                    """Push subprocess stdout lines into a thread-safe queue."""
                    try:
                        for line in stream:
                            out_queue.put(line)
                    finally:
                        out_queue.put(None)

                reader_thread = threading.Thread(target=_reader, daemon=True)
                reader_thread.start()

                deadline = time.monotonic() + timeout
                timed_out = False

                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        print(
                            f"  {phase_name}{where}: timeout reached ({timeout}s), "
                            "sending SIGKILL to pip process.",
                            file=sys.stderr,
                        )
                        proc.kill()
                        break
                    try:
                        item = line_queue.get(timeout=min(0.5, max(0.01, remaining)))
                    except queue.Empty:
                        if proc.poll() is not None and not reader_thread.is_alive():
                            break
                        continue

                    if item is None:
                        if proc.poll() is not None:
                            break
                        continue

                    out_lines.append(item)
                    print(f"    {attempt_prefix} {sanitize_log_line(item.rstrip())}")

                # Drain any remaining buffered output after process exit/kill.
                while True:
                    try:
                        leftover = line_queue.get_nowait()
                    except queue.Empty:
                        break
                    if leftover is None:
                        continue
                    out_lines.append(leftover)
                    print(
                        f"    {attempt_prefix} {sanitize_log_line(leftover.rstrip())}"
                    )

                if timed_out:
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(
                            f"  {phase_name}{where}: process did not exit after "
                            "SIGKILL; retrying SIGKILL before surfacing timeout.",
                            file=sys.stderr,
                        )
                        proc.kill()
                        proc.wait(timeout=5)
                    raise subprocess.TimeoutExpired(cmd=cmd_args, timeout=timeout)

                if proc.poll() is None:
                    return_code = proc.wait(timeout=5)
                else:
                    return_code = proc.returncode
        except subprocess.TimeoutExpired:
            stdout_tail = ("".join(out_lines).strip())[-800:]
            last_error = RuntimeError(
                f"{phase_name}{where} timed out after {timeout}s."
                "\n"
                f"stdout(last): {stdout_tail or '<empty>'}\n"
                "stderr(last): <merged-into-stdout>"
            )
        else:
            stdout_text = "".join(out_lines).strip()
            if return_code == 0:
                return subprocess.CompletedProcess(
                    args=cmd_args,
                    returncode=return_code,
                    stdout="".join(out_lines),
                    stderr="",
                )
            failed_req = _extract_failed_requirement(stdout_text)
            req_note = f" Suspected requirement: {failed_req}." if failed_req else ""
            last_error = RuntimeError(
                f"{phase_name}{where} failed with exit code {return_code}.{req_note}\n"
                f"stdout(last): {(stdout_text[-800:] or '<empty>')}\n"
                "stderr(last): <merged-into-stdout>"
            )
        finally:
            if proc_registry is not None and proc is not None:
                procs_list, procs_lock = proc_registry
                with procs_lock:
                    try:
                        procs_list.remove(proc)
                    except ValueError:
                        pass

        if attempt < attempts:
            print(
                f"  {phase_name}{where} failed on attempt {attempt}/{attempts}; "
                f"retrying in {retry_backoff_sec}s ...",
                file=sys.stderr,
            )
            if retry_backoff_sec > 0:
                time.sleep(retry_backoff_sec)

    assert last_error is not None
    raise last_error


def generate_for_index(
    index_url: str | None,
    root_names: list[str],
    root_bare_names: list[str] | None = None,
    platform_groups: list[list[str]] | None = None,
    out_path: Path = Path("requirements.txt"),
    dry_run: bool = False,
    download_timeout: int = 480,
) -> int:
    """Generate a fully pinned requirements file for one package index.

    Phase 1 resolves dependency versions using ``pip install --dry-run --report``.
    Phase 2 downloads artifacts for each platform group in parallel and computes
    SHA256 hashes, then writes atomic output.

    Args:
        index_url: Explicit package index URL. If empty/``None``, use system pip.
        root_names: Root packages to resolve (may include ``==version`` pins).
        root_bare_names: Root package names without version pins, used for
            output ordering.  Defaults to ``root_names`` when ``None``.
        platform_groups: Platform tag groups for parallel download passes.
        out_path: Destination requirements file path.
        dry_run: If ``True``, print commands only and do not execute pip.
        download_timeout: Per-platform-group download timeout in seconds.

    Returns:
        ``0`` on success, :data:`SKIP_NOT_AVAILABLE` when a root package
        version is not yet on the index, ``1`` on any other failure.
    """
    if root_bare_names is None:
        root_bare_names = root_names
    if platform_groups is None:
        platform_groups = [resolve_platform(DEFAULT_PLATFORM)]
    pip_ok, pip_msg = _pip_supports_report()
    if not pip_ok:
        print(f"Error: {pip_msg}", file=sys.stderr)
        return 1

    use_system_index = index_url is None or (
        isinstance(index_url, str) and not index_url.strip()
    )
    if use_system_index:
        index_url = get_system_index_url()
        if index_url:
            print(f"  Using system pip index: {redact_index_url(index_url)}")
        else:
            print("  Using system pip config (no explicit index URL found)")

    with tempfile.TemporaryDirectory(prefix="mlserver-req-") as tmp:
        resolve_dir = Path(tmp) / "resolve"
        download_dir = Path(tmp) / "wheels"
        resolve_dir.mkdir()
        download_dir.mkdir()

        req_roots_lines = ([f"--index-url={index_url}", ""] if index_url else []) + [
            name for name in root_names
        ]
        req_roots = Path(tmp) / "req_roots.txt"
        req_roots.write_text("\n".join(req_roots_lines) + "\n")

        print("  Phase 1: Resolving dependency tree from index ...")
        resolve_report = resolve_dir / "resolve-report.json"
        resolve_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--report",
            resolve_report,
            "-r",
            req_roots,
        ]
        if index_url:
            resolve_cmd.extend(["--index-url", index_url])
        if dry_run:
            print("  Would run: " + format_command_for_log(resolve_cmd))
            for group in platform_groups:
                cmd = [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    "-r",
                    Path(tmp) / "req_all.txt",
                    "-d",
                    download_dir / f"group-{'-'.join(group)}",
                    "--no-deps",
                ]
                if index_url:
                    cmd.extend(["--index-url", index_url])
                for p in group:
                    cmd.extend(["--platform", p])
                print("  Would run: " + format_command_for_log(cmd))
            return 0
        try:
            run_pip_command(
                resolve_cmd,
                timeout=300,
                phase_name="Phase 1",
                attempts=2,
                retry_backoff_sec=5,
            )
        except RuntimeError as e:
            error_text = str(e)
            not_found_match = re.search(
                r"No matching distribution found for\s+(\S+)",
                error_text,
                re.IGNORECASE,
            )
            if not_found_match:
                failed_req = not_found_match.group(1)
                failed_name = normalize_distribution_name(
                    failed_req.split("[", 1)[0].split("==", 1)[0]
                )
                root_norm = {
                    normalize_distribution_name(n.split("[", 1)[0])
                    for n in root_bare_names
                }
                if failed_name in root_norm:
                    print(
                        f"  Root package {failed_req} not yet available "
                        "on index. Skipping regeneration.",
                        file=sys.stderr,
                    )
                    return SKIP_NOT_AVAILABLE
            print(f"  {e}", file=sys.stderr)
            return 1

        hash_cache: dict[str, str] = {}
        resolved, _ = parse_report_packages_and_hashes(
            report_path=resolve_report,
            download_dir=resolve_dir,
            hash_cache=hash_cache,
        )
        print(f"  Resolved {len(resolved)} packages.")

        req_all_lines = ([f"--index-url={index_url}", ""] if index_url else []) + [
            f"{n}=={v}" for n, v in resolved
        ]
        req_all = Path(tmp) / "req_all.txt"
        req_all.write_text("\n".join(req_all_lines) + "\n")

        active_procs: list[subprocess.Popen] = []
        active_procs_lock = threading.Lock()
        registry = (active_procs, active_procs_lock)

        def download_for_group(
            group_index: int, group: list[str]
        ) -> tuple[list[str], Path]:
            """Download resolved artifacts for a single platform group.

            Args:
                group_index: Numeric index used for deterministic temp directory.
                group: Platform tags for this download pass.

            Returns:
                The input platform group and its download directory path.
            """
            group_name = ",".join(group)
            group_dir = download_dir / f"group-{group_index}"
            group_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "download",
                "-r",
                req_all,
                "-d",
                group_dir,
                "--no-deps",
            ]
            if index_url:
                cmd.extend(["--index-url", index_url])
            for p in group:
                cmd.extend(["--platform", p])
            print(f"  Phase 2: Downloading for {group_name} ...")
            run_pip_command(
                cmd,
                timeout=download_timeout,
                phase_name="Phase 2",
                context=f"platforms={group_name}",
                attempts=2,
                retry_backoff_sec=10,
                proc_registry=registry,
            )
            return group, group_dir

        hashes_by_package_sets: dict[tuple[str, str], set[str]] = {}
        max_workers = max(1, min(len(platform_groups), os.cpu_count() or 1))
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        prev_sigterm = signal.getsignal(signal.SIGTERM)
        prev_sigint = signal.getsignal(signal.SIGINT)

        def _cleanup_handler(signum, frame):
            _kill_registered_procs(active_procs, active_procs_lock)
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, _cleanup_handler)
        signal.signal(signal.SIGINT, _cleanup_handler)

        try:
            futures = {
                executor.submit(download_for_group, idx, group): group
                for idx, group in enumerate(platform_groups)
            }
            failures: list[tuple[list[str], RuntimeError]] = []
            for future in concurrent.futures.as_completed(futures):
                group = futures[future]
                try:
                    _, group_dir = future.result()
                except RuntimeError as e:
                    failures.append((group, e))
                    continue
                group_hashes = collect_hashes_from_download_dir(group_dir, hash_cache)
                for pkg, hashes in group_hashes.items():
                    hashes_by_package_sets.setdefault(pkg, set()).update(hashes)

            if failures:
                for group, err in failures:
                    print(f"  Phase 2 failed for {group}: {err}", file=sys.stderr)
                return 1
        finally:
            executor.shutdown(wait=True)
            signal.signal(signal.SIGTERM, prev_sigterm)
            signal.signal(signal.SIGINT, prev_sigint)

        hashes_by_package = {
            key: sorted(values) for key, values in hashes_by_package_sets.items()
        }

    resolved_by_norm: dict[str, tuple[str, str]] = {
        normalize_distribution_name(n): (n, v) for n, v in resolved
    }
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for root_name in root_bare_names:
        bare_name = root_name.split("[", 1)[0]
        nv = resolved_by_norm.get(normalize_distribution_name(bare_name))
        if nv and (normalize_distribution_name(nv[0]), nv[1]) not in seen:
            seen.add((normalize_distribution_name(nv[0]), nv[1]))
            ordered.append(nv)
    remaining: list[tuple[str, str]] = []
    for n, v in resolved:
        key = (normalize_distribution_name(n), v)
        if key not in seen:
            seen.add(key)
            remaining.append((n, v))
    ordered.extend(
        sorted(
            remaining, key=lambda item: (normalize_distribution_name(item[0]), item[1])
        )
    )

    lines: list[str] = []
    if index_url:
        lines.append(f"--index-url={redact_index_url(index_url)}")
        lines.append("")
    missing_hashes: list[str] = []
    for name, version in ordered:
        key = (normalize_distribution_name(name), version)
        hashes_list = hashes_by_package.get(key)
        if not hashes_list:
            missing_hashes.append(f"{name}=={version}")
            continue
        line0 = f"{name}=={version} \\"
        lines.append(line0)
        for i, h in enumerate(hashes_list):
            suffix = " \\" if i < len(hashes_list) - 1 else ""
            lines.append(f"    --hash=sha256:{h}{suffix}")
        lines.append("")

    if missing_hashes:
        print(
            "  Error: missing hashes for: " + ", ".join(sorted(missing_hashes)),
            file=sys.stderr,
        )
        return 1

    temp_output = out_path.with_suffix(out_path.suffix + ".tmp")
    temp_output.write_text("\n".join(lines) + "\n")
    temp_output.replace(out_path)
    print(f"  Wrote {len(ordered)} packages to {out_path}")
    return 0


def main() -> int:
    """CLI entry point for requirements generation and helper operations.

    Returns:
        Process-style exit code (``0`` success, ``1`` failure).
    """
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    try:
        config = load_config(script_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(
        description=(
            "Generate pinned (hashed) requirements from root packages "
            "(in base image)."
        )
    )
    parser.add_argument(
        "--print-base-image",
        metavar="DOCKERFILE",
        dest="print_base_image",
        default=None,
        help="Print base image from Dockerfile and exit (for CI).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        dest="output",
        help="Output path (default: requirements.txt in current directory)",
    )
    parser.add_argument(
        "--platform",
        action="append",
        default=[],
        dest="platforms",
        help=(
            "Platform to generate for (can repeat). "
            "Supported: x86_64, amd64, aarch64, arm64, ppc64le, s390x. "
            "Aliases: amd64 -> x86_64, arm64 -> aarch64. "
            "Overrides config 'platforms' when set. Default: x86_64 only."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        dest="timeout",
        help=(
            "Download timeout per platform group in seconds. "
            "Overrides config 'download_timeout'. Default: 480."
        ),
    )
    parser.add_argument(
        "--index-url",
        default=None,
        dest="index_url",
        help=(
            "Explicit package index URL to use for resolve/download. "
            "Default: use system pip config in current environment."
        ),
    )
    parser.add_argument(
        "--variant",
        default=None,
        dest="variant",
        help=(
            "Select a variant from requirements-config.json. "
            "Each variant must define its own root_packages."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print pip download commands, do not run",
    )
    args = parser.parse_args()

    if args.print_base_image:
        try:
            image = get_base_image_from_dockerfile(repo_root, args.print_base_image)
            print(image)
            return 0
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    if args.timeout is not None and args.timeout <= 0:
        print("Error: --timeout must be a positive integer.", file=sys.stderr)
        return 1

    if args.output is not None:
        out_path = args.output.resolve()
    else:
        out_path = (Path.cwd() / "requirements.txt").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    variants = config.get("variants", [])
    if not args.variant:
        available = [v.get("name", "?") for v in variants]
        print(
            f"Error: --variant is required. Available: {available}",
            file=sys.stderr,
        )
        return 1
    matched = [v for v in variants if v.get("name") == args.variant]
    if not matched:
        available = [v.get("name", "?") for v in variants]
        print(
            f"Error: variant '{args.variant}' not found. Available: {available}",
            file=sys.stderr,
        )
        return 1
    variant = matched[0]
    raw_root_packages = variant.get("root_packages", [])
    if not raw_root_packages:
        print(
            f"Error: variant '{args.variant}' has no root_packages.",
            file=sys.stderr,
        )
        return 1
    root_names, root_bare_names = build_requirement_strings(raw_root_packages)

    try:
        platform_groups, platform_source = resolve_platform_groups(
            args.platforms, variant
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Platforms ({platform_source}): " f"{[g[0] for g in platform_groups]}")

    download_timeout = resolve_download_timeout(args.timeout, variant)

    print(f"Root packages (pinned): {', '.join(root_names)}")
    if args.index_url:
        print(f"Output -> {out_path} (index: {redact_index_url(args.index_url)})")
    else:
        print(f"Output -> {out_path} (system pip)")
    result = generate_for_index(
        index_url=args.index_url,
        root_names=root_names,
        root_bare_names=root_bare_names,
        platform_groups=platform_groups,
        out_path=out_path,
        dry_run=args.dry_run,
        download_timeout=download_timeout,
    )
    if result == SKIP_NOT_AVAILABLE:
        skip_info = {
            "variant": args.variant,
            "expected_packages": root_names,
            "reason": "Root package version not yet available on index",
        }
        skip_marker = out_path.parent / f".skip-{args.variant}.json"
        skip_marker.write_text(json.dumps(skip_info, indent=2) + "\n")
        print(f"  Skip marker written to {skip_marker}")
        return 0
    return result


if __name__ == "__main__":
    raise SystemExit(main())
