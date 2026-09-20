"""Offline asset verification performed before the vLLM process starts."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .baseline_config import AssetConfig
from .baseline_manifest import (
    FilePin,
    load_control_manifest,
    validate_wheelhouse_checksum_manifest,
    verify_pinned_files,
)


class PreflightReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    control_id: str
    manifest_path: Path
    model_dir: Path
    wheelhouse_dir: Path
    full_verification: bool
    verified_files: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


class AssetPreflightError(RuntimeError):
    """Pinned offline inputs are missing or have changed."""

    def __init__(self, report: PreflightReport) -> None:
        self.report = report
        details = "\n".join(f"- {error}" for error in report.errors)
        super().__init__(f"offline asset preflight failed:\n{details}")


def _verify_presence_and_size(root: Path, pins: tuple[FilePin, ...]) -> list[str]:
    errors: list[str] = []
    for pin in pins:
        path = root / pin.name
        if not path.is_file():
            errors.append(f"missing pinned file: {path}")
        elif pin.size_bytes is not None and path.stat().st_size != pin.size_bytes:
            errors.append(
                f"size mismatch for {path}: expected {pin.size_bytes}, "
                f"got {path.stat().st_size}"
            )
    return errors


def run_asset_preflight(config: AssetConfig) -> PreflightReport:
    """Verify the mounted model and wheelhouse without downloading anything."""

    manifest_path = config.manifest_path.resolve()
    model_dir = config.model_dir.resolve()
    wheelhouse_dir = config.wheelhouse_dir.resolve()
    manifest = load_control_manifest(manifest_path)
    errors: list[str] = []
    verified: list[str] = []

    if not model_dir.is_dir():
        errors.append(f"model directory does not exist: {model_dir}")
    if not wheelhouse_dir.is_dir():
        errors.append(f"wheelhouse directory does not exist: {wheelhouse_dir}")

    checksum_path = wheelhouse_dir / manifest.wheelhouse.checksum_manifest.name
    if wheelhouse_dir.is_dir():
        checksum_errors = verify_pinned_files(
            wheelhouse_dir, (manifest.wheelhouse.checksum_manifest,)
        )
        errors.extend(checksum_errors)
        if not checksum_errors:
            verified.append(str(checksum_path))
            try:
                checksum_text = checksum_path.read_text(encoding="utf-8")
                errors.extend(
                    validate_wheelhouse_checksum_manifest(manifest, checksum_text)
                )
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                errors.append(f"invalid wheelhouse checksum manifest: {exc}")

        for pin in manifest.wheelhouse.packages:
            wheel_errors = verify_pinned_files(wheelhouse_dir, (pin,))
            errors.extend(wheel_errors)
            if not wheel_errors:
                verified.append(str(wheelhouse_dir / pin.name))

    if model_dir.is_dir():
        for pin in manifest.model.files:
            if config.full_verification:
                model_errors = verify_pinned_files(model_dir, (pin,))
            else:
                model_errors = _verify_presence_and_size(model_dir, (pin,))
            errors.extend(model_errors)
            if not model_errors:
                verified.append(str(model_dir / pin.name))

    return PreflightReport(
        control_id=manifest.control_id,
        manifest_path=manifest_path,
        model_dir=model_dir,
        wheelhouse_dir=wheelhouse_dir,
        full_verification=config.full_verification,
        verified_files=tuple(verified),
        errors=tuple(errors),
    )


def require_asset_preflight(config: AssetConfig) -> PreflightReport:
    report = run_asset_preflight(config)
    if not report.ok:
        raise AssetPreflightError(report)
    return report
