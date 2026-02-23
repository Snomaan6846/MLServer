#!/usr/bin/env python3
"""
Generate requirements-<name>.txt with pinned versions and SHA256 hashes
for x86_64 and aarch64.

Run inside the base image container for each variant; the image's pip index is used.
Root packages are resolved as latest from the index (no version file). Output is
pinned for reproducible installs.

Options:
  -o PATH              Output path (default: requirements.txt in cwd).
  --print-base-image   Print base image from Dockerfile and exit (for CI).

Usage (in container):
  python hack/generate-pinned-requirements.py -o requirements/requirements-cpu.txt
Usage (on host):
  python hack/generate-pinned-requirements.py --print-base-image Dockerfile.konflux
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import BadZipFile, ZipFile


# Phase 2: a pip download per arch; multiple tags per arch for index compatibility.
DEFAULT_PLATFORMS = [
    ["manylinux2014_x86_64", "manylinux_2_34_x86_64", "linux_x86_64"],
    ["manylinux2014_aarch64", "manylinux_2_34_aarch64", "linux_aarch64"],
]

CONFIG_FILENAME = "requirements-config.json"


def normalize_distribution_name(name: str) -> str:
    """PEP 503 canonical name: lowercase, underscores and dots to hyphens."""
    return name.lower().replace("_", "-").replace(".", "-")


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
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
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
    r"^\s*ARG\s+(\w+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|(\S+))", re.IGNORECASE
)
_FROM_RE = re.compile(r"^\s*FROM\s+(.+)$", re.IGNORECASE)


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
        arg_m = _ARG_RE.match(line)
        if arg_m:
            name, dq, sq, unq = arg_m.groups()
            value = dq or sq or (unq or "").strip()
            if value:
                args[name] = value
            continue
        from_m = _FROM_RE.match(line)
        if from_m:
            from_line = from_m.group(1).strip()
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
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_name_version_from_wheel(wheel_path: Path) -> tuple[str, str] | None:
    """Read METADATA from wheel for Name and Version. Returns (name, version)."""
    try:
        with ZipFile(wheel_path) as z:
            for info in z.infolist():
                if info.filename.endswith(".dist-info/METADATA"):
                    with z.open(info) as f:
                        content = f.read().decode("utf-8", errors="replace")
                    name = version = None
                    for line in content.splitlines():
                        if line.startswith("Name:"):
                            name = normalize_distribution_name(line[5:].strip())
                        elif line.startswith("Version:"):
                            version = line[8:].strip()
                        if name and version:
                            return (name, version)
                    return (name, version) if name and version else None
    except (OSError, BadZipFile):
        pass
    return None


def get_name_version_from_sdist(path: Path) -> tuple[str, str] | None:
    """Parse sdist filename to get (name, version). E.g. my_pkg-1.0.0.tar.gz."""
    name = path.name
    for suffix in (".tar.gz", ".tar.bz2", ".zip", ".tar.xz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    else:
        return None
    for i in range(len(name) - 1, -1, -1):
        if name[i] == "-" and i + 1 < len(name) and name[i + 1].isdigit():
            dist = normalize_distribution_name(name[:i])
            ver = name[i + 1 :]
            if dist and ver:
                return (dist, ver)
    return None


def get_name_version(path: Path) -> tuple[str, str] | None:
    """Return (name, version) from a wheel or sdist path."""
    if path.suffix == ".whl":
        return get_name_version_from_wheel(path)
    return get_name_version_from_sdist(path)


def collect_packages_and_hashes(
    download_dir: Path,
) -> dict[tuple[str, str], list[str]]:
    """For each wheel/sdist get (name, version) and hash. Return map to hashes."""
    result: dict[tuple[str, str], list[str]] = {}
    for path in download_dir.iterdir():
        nv = get_name_version(path)
        if nv:
            h = sha256_file(path)
            result.setdefault(nv, []).append(h)
    for k in result:
        result[k] = sorted(set(result[k]))
    return result


def collect_resolved_versions(download_dir: Path) -> list[tuple[str, str]]:
    """Return (name, version) list from artifacts in stable order (roots first)."""
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for path in sorted(download_dir.iterdir()):
        nv = get_name_version(path)
        if nv and nv not in seen:
            seen.add(nv)
            pairs.append(nv)
    return pairs


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
    use_system_index = index_url is None or (
        isinstance(index_url, str) and not index_url.strip()
    )
    if use_system_index:
        index_url = get_system_index_url()
        if index_url:
            print(f"  Using system pip index: {index_url}")
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
        resolve_cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "-r",
            str(req_roots),
            "-d",
            str(resolve_dir),
        ]
        if index_url:
            resolve_cmd.extend(["--index-url", index_url])
        if dry_run:
            print(f"  Would run: {' '.join(resolve_cmd)}")
            return 0
        try:
            subprocess.run(resolve_cmd, check=True, timeout=600)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  Phase 1 failed: {e}", file=sys.stderr)
            return 1

        resolved = collect_resolved_versions(resolve_dir)
        print(f"  Resolved {len(resolved)} packages.")

        req_all_lines = ([f"--index-url={index_url}", ""] if index_url else []) + [
            f"{n}=={v}" for n, v in resolved
        ]
        req_all = Path(tmp) / "req_all.txt"
        req_all.write_text("\n".join(req_all_lines) + "\n")

        for group in platform_groups:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "download",
                "-r",
                str(req_all),
                "-d",
                str(download_dir),
                "--no-deps",
            ]
            if index_url:
                cmd.extend(["--index-url", index_url])
            for p in group:
                cmd.extend(["--platform", p])
            print(f"  Phase 2: Downloading for {', '.join(group)} ...")
            try:
                subprocess.run(cmd, check=True, timeout=600)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"  Phase 2 failed for {group}: {e}", file=sys.stderr)
                return 1

        hashes_by_package = collect_packages_and_hashes(download_dir)

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
    for key in sorted(hashes_by_package.keys()):
        if key not in seen:
            seen.add(key)
            ordered.append((key[0], key[1]))

    lines: list[str] = []
    if index_url:
        lines.append(f"--index-url={index_url}")
        lines.append("")
    for name, version in ordered:
        key = (normalize_distribution_name(name), version)
        hashes_list = hashes_by_package.get(key)
        if not hashes_list:
            print(f"  Warning: no hashes for {name}=={version}", file=sys.stderr)
            lines.append(f"{name}=={version}")
            lines.append("")
            continue
        line0 = f"{name}=={version} \\"
        lines.append(line0)
        for i, h in enumerate(hashes_list):
            suffix = " \\" if i < len(hashes_list) - 1 else ""
            lines.append(f"    --hash=sha256:{h}{suffix}")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n")
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
    print(f"Output -> {out_path} (system pip)")
    return generate_for_index(
        index_url=None,
        root_names=root_names,
        platform_groups=platform_groups,
        out_path=out_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
