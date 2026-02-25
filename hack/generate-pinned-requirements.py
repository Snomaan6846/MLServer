#!/usr/bin/env python3
"""
Generate requirements-<name>.txt with pinned versions and SHA256 hashes
for x86_64 and aarch64.

Run inside the base image container for each variant; the image's pip index is used.
Root packages are resolved as latest from the index (no version file). Output is
pinned for reproducible installs.

Options:
  -o PATH              Output path (default: requirements.txt in cwd).
  --index-url URL      Override package index URL (otherwise use system pip config).
  --print-base-image   Print base image from Dockerfile and exit (for CI).

Usage (in container):
  python hack/generate-pinned-requirements.py -o requirements/requirements-cpu.txt
Usage (on host):
  python hack/generate-pinned-requirements.py --print-base-image Dockerfile.konflux
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urlparse


# Phase 2: a pip download per arch; multiple tags per arch for index compatibility.
DEFAULT_PLATFORMS = [
    ["manylinux2014_x86_64", "manylinux_2_34_x86_64", "linux_x86_64"],
    ["manylinux2014_aarch64", "manylinux_2_34_aarch64", "linux_aarch64"],
]

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
    """PEP 503 canonical name: lowercase, underscores and dots to hyphens."""
    return name.lower().replace("_", "-").replace(".", "-")


def redact_index_url(url: str) -> str:
    """
    Redact sensitive components from index URL for logs/output files.
    Keeps scheme/host/path but removes userinfo and sensitive query params.
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


def get_system_index_url() -> str | None:
    """Get pip index URL from env or pip config (for use inside container)."""
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
    """Load variants (name + dockerfile) and root_packages from config."""
    config_path = script_dir / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    data = json.loads(config_path.read_text())
    if "root_packages" not in data:
        raise ValueError("Config missing required key: 'root_packages'")
    if "variants" not in data:
        if "indexes" in data:
            # Backward compatibility: accept legacy "indexes" key.
            data["variants"] = data["indexes"]
        else:
            raise ValueError("Config missing required key: 'variants'")
    if not isinstance(data["root_packages"], list) or not data["root_packages"]:
        raise ValueError("Config 'root_packages' must be a non-empty list")
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
    """Strip comments from a Dockerfile line."""
    return line.split("#", 1)[0].strip()


def get_base_image_from_dockerfile(repo_root: Path, dockerfile_path: str) -> str:
    """
    Parse the Dockerfile at repo_root/dockerfile_path and return the base image (FROM).
    Resolves ARG defaults so that FROM ${RUNTIME_BASE_IMAGE} is expanded using
    ARG RUNTIME_BASE_IMAGE="...".
    """
    df_path = (repo_root / dockerfile_path).resolve()
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
        var = m.group(1)
        return args.get(var, m.group(0))

    resolved = re.sub(r"\$\{(\w+)\}", repl, from_line)
    if "${" in resolved:
        raise ValueError(f"Unresolved variable(s) in FROM in {df_path}: {from_line}")
    return resolved.strip()


def sha256_file(path: Path) -> str:
    """Return SHA256 hash of file as hex string."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_report(report_path: Path) -> dict:
    data = json.loads(report_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid pip report format in {report_path}")
    return data


def _report_filename_for_item(item: dict) -> str | None:
    download_info = item.get("download_info") or {}
    if not isinstance(download_info, dict):
        return None
    url = download_info.get("url")
    if not isinstance(url, str) or not url:
        return None
    return Path(unquote(urlparse(url).path)).name or None


def _name_version_from_filename(path: Path) -> tuple[str, str] | None:
    """Best-effort (name, version) extraction from wheel/sdist filename."""
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
    """Collect hashes by parsing downloaded artifact filenames (fail-fast)."""
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
    """
    Parse pip report once and return both package list and hash map.
    Prefer report-provided hashes and hash local files only when needed.
    """
    hash_cache = hash_cache if hash_cache is not None else {}
    report = _load_report(report_path)
    install_items = report.get("install")
    if not isinstance(install_items, list):
        raise ValueError(f"Invalid pip report: missing/invalid 'install' list in {report_path}")
    seen: set[tuple[str, str]] = set()
    packages: list[tuple[str, str]] = []
    hashes_by_package: dict[tuple[str, str], set[str]] = {}
    for idx, item in enumerate(install_items):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid pip report item at install[{idx}]: expected object")
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
    """Return (supported, message). pip --report is supported in pip >= 22.2."""
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
) -> subprocess.CompletedProcess:
    """Run pip command with live output and rich diagnostics."""
    where = f" ({context})" if context else ""
    phase_tag = f"[{phase_name}]"
    context_tag = f" [{context}]" if context else ""
    prefix = f"{phase_tag}{context_tag}"
    out_lines: list[str] = []
    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        ) as proc:
            assert proc.stdout is not None
            while True:
                line = proc.stdout.readline()
                if line:
                    out_lines.append(line)
                    print(f"    {prefix} {line.rstrip()}")
                    continue
                if proc.poll() is not None:
                    break
            return_code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        stdout_tail = ("".join(out_lines).strip())[-800:]
        raise RuntimeError(
            f"{phase_name}{where} timed out after {timeout}s."
            "\n"
            f"stdout(last): {stdout_tail or '<empty>'}\n"
            "stderr(last): <merged-into-stdout>"
        ) from e
    stdout_text = "".join(out_lines).strip()
    if return_code != 0:
        failed_req = _extract_failed_requirement(stdout_text)
        req_note = f" Suspected requirement: {failed_req}." if failed_req else ""
        raise RuntimeError(
            f"{phase_name}{where} failed with exit code {return_code}.{req_note}\n"
            f"stdout(last): {(stdout_text[-800:] or '<empty>')}\n"
            "stderr(last): <merged-into-stdout>"
        )
    return subprocess.CompletedProcess(
        args=[str(x) for x in cmd],
        returncode=return_code,
        stdout="".join(out_lines),
        stderr="",
    )


def generate_for_index(
    index_url: str | None,
    root_names: list[str],
    platform_groups: list[list[str]],
    out_path: Path,
    dry_run: bool = False,
) -> int:
    """Run Phase 1 + Phase 2 and write hashed requirements to out_path.
    Root packages are resolved as latest from the index. If index_url is None,
    use system pip. Returns 0 on success."""
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

        print("  Phase 1: Resolving dependency tree from index (latest) ...")
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
            print("  Would run: " + " ".join(str(x) for x in resolve_cmd))
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
                print("  Would run: " + " ".join(str(x) for x in cmd))
            return 0
        try:
            run_pip_command(resolve_cmd, timeout=600, phase_name="Phase 1")
        except RuntimeError as e:
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

        def download_for_group(
            group_index: int, group: list[str]
        ) -> tuple[list[str], Path]:
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
                timeout=600,
                phase_name="Phase 2",
                context=f"platforms={group_name}",
            )
            return group, group_dir

        hashes_by_package_sets: dict[tuple[str, str], set[str]] = {}
        max_workers = max(1, min(len(platform_groups), os.cpu_count() or 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_for_group, idx, group): group
                for idx, group in enumerate(platform_groups)
            }
            for future in concurrent.futures.as_completed(futures):
                group = futures[future]
                try:
                    _, group_dir = future.result()
                except RuntimeError as e:
                    print(f"  {e}", file=sys.stderr)
                    print(f"  Phase 2 failed for {group}", file=sys.stderr)
                    return 1
                group_hashes = collect_hashes_from_download_dir(group_dir, hash_cache)
                for pkg, hashes in group_hashes.items():
                    hashes_by_package_sets.setdefault(pkg, set()).update(hashes)

        hashes_by_package = {
            key: sorted(values) for key, values in hashes_by_package_sets.items()
        }

    resolved_by_norm: dict[str, tuple[str, str]] = {
        normalize_distribution_name(n): (n, v) for n, v in resolved
    }
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for root_name in root_names:
        nv = resolved_by_norm.get(normalize_distribution_name(root_name))
        if nv and (normalize_distribution_name(nv[0]), nv[1]) not in seen:
            seen.add((normalize_distribution_name(nv[0]), nv[1]))
            ordered.append(nv)
    for n, v in resolved:
        key = (normalize_distribution_name(n), v)
        if key not in seen:
            seen.add(key)
            ordered.append((n, v))

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
        help="Platform tag; can repeat. Default: manylinux2014+linux x86_64/aarch64",
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

    if args.output is not None:
        out_path = args.output.resolve()
    else:
        out_path = (Path.cwd() / "requirements.txt").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.platforms:
        platform_groups = [[p] for p in args.platforms]
    else:
        platform_groups = DEFAULT_PLATFORMS

    root_names = config["root_packages"]
    print(f"Root packages (latest from index): {', '.join(root_names)}")
    if args.index_url:
        print(f"Output -> {out_path} (index: {redact_index_url(args.index_url)})")
    else:
        print(f"Output -> {out_path} (system pip)")
    return generate_for_index(
        index_url=args.index_url,
        root_names=root_names,
        platform_groups=platform_groups,
        out_path=out_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
