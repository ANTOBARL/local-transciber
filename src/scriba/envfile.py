"""Read/write a managed block inside a dotenv file, leaving everything else untouched."""

from __future__ import annotations

import os
import re
from pathlib import Path

BLOCK_START = "# >>> scriba optimizer"
BLOCK_END = "# <<< scriba optimizer"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def env_file_path() -> Path:
    """Dotenv file used for tuned settings.

    SCRIBA_ENV_FILE wins (used by the Docker image); otherwise the project .env for
    source/editable installs, falling back to the current directory.
    """
    explicit = os.environ.get("SCRIBA_ENV_FILE")
    if explicit:
        return Path(explicit)
    root = project_root()
    if (root / "pyproject.toml").is_file():
        return root / ".env"
    return Path.cwd() / ".env"


def env_files_to_read() -> list[Path]:
    """Files read into settings, lowest precedence first."""
    primary = env_file_path()
    files = [primary]
    cwd_env = Path.cwd() / ".env"
    if not os.environ.get("SCRIBA_ENV_FILE") and cwd_env.resolve() != primary.resolve():
        files.append(cwd_env)
    return files


def read_managed_block(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    match = re.search(rf"{re.escape(BLOCK_START)}.*?\n(.*?){re.escape(BLOCK_END)}", text, flags=re.S)
    if not match:
        return {}
    values = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def write_managed_block(path: Path, values: dict[str, object], comment_lines: list[str] | None = None) -> Path:
    """Replace (or append) the optimizer block. Other lines, e.g. tokens, are preserved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    original = path.read_text(encoding="utf-8-sig") if path.is_file() else ""
    body = [BLOCK_START + " (generated - re-run the optimizer to update)"]
    body += [f"# {line}" for line in (comment_lines or [])]
    body += [f"{key}={value}" for key, value in values.items()]
    body.append(BLOCK_END)
    block = "\n".join(body) + "\n"

    pattern = re.compile(rf"{re.escape(BLOCK_START)}.*?{re.escape(BLOCK_END)}\n?", flags=re.S)
    if pattern.search(original):
        updated = pattern.sub(lambda _: block, original, count=1)
    else:
        separator = "" if not original or original.endswith("\n\n") else ("\n" if original.endswith("\n") else "\n\n")
        updated = original + separator + block

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(updated, encoding="utf-8", newline="\n")
    tmp.replace(path)
    return path
