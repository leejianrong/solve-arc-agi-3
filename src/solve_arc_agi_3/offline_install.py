"""Offline-only installation command for the verified vLLM wheelhouse."""

from __future__ import annotations

from pathlib import Path


def build_offline_install_command(
    wheelhouse_dir: Path, *, python_executable: str = "python"
) -> tuple[str, ...]:
    """Build a pip command that cannot consult a package index."""

    root = wheelhouse_dir.resolve()
    return (
        python_executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-index",
        "--find-links",
        str(root),
        "-r",
        str(root / "requirements.lock"),
    )
