import hashlib
from pathlib import Path

from solve_arc_agi_3.baseline_config import AssetConfig
from solve_arc_agi_3.baseline_manifest import FilePin, load_control_manifest
from solve_arc_agi_3.preflight import run_asset_preflight

PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_MANIFEST = PROJECT_ROOT / "configs/baselines/duck-control-v1.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _asset_fixture(tmp_path: Path) -> AssetConfig:
    model_dir = tmp_path / "model"
    wheelhouse_dir = tmp_path / "wheels"
    model_dir.mkdir()
    wheelhouse_dir.mkdir()
    model_bytes = b"model"
    tokenizer_bytes = b"tokenizer"
    wheel_bytes = b"wheel"
    (model_dir / "model.safetensors").write_bytes(model_bytes)
    (model_dir / "tokenizer.json").write_bytes(tokenizer_bytes)
    (wheelhouse_dir / "core.whl").write_bytes(wheel_bytes)

    package_pin = FilePin(name="core.whl", sha256=_sha(wheel_bytes))
    checksum_bytes = f"{package_pin.sha256}  {package_pin.name}\n".encode()
    (wheelhouse_dir / "SHA256SUMS").write_bytes(checksum_bytes)

    source = load_control_manifest(SOURCE_MANIFEST)
    manifest = source.model_copy(
        update={
            "model": source.model.model_copy(
                update={
                    "files": (
                        FilePin(
                            name="model.safetensors",
                            sha256=_sha(model_bytes),
                            size_bytes=len(model_bytes),
                        ),
                        FilePin(
                            name="tokenizer.json",
                            sha256=_sha(tokenizer_bytes),
                            size_bytes=len(tokenizer_bytes),
                        ),
                    )
                }
            ),
            "wheelhouse": source.wheelhouse.model_copy(
                update={
                    "checksum_manifest": FilePin(
                        name="SHA256SUMS",
                        sha256=_sha(checksum_bytes),
                        size_bytes=len(checksum_bytes),
                    ),
                    "checksum_entry_count": 1,
                    "packages": (package_pin,),
                }
            ),
        }
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return AssetConfig(
        manifest_path=manifest_path,
        model_dir=model_dir,
        wheelhouse_dir=wheelhouse_dir,
        full_verification=True,
        offline=True,
    )


def test_full_preflight_verifies_checksum_wheels_model_and_tokenizer(
    tmp_path: Path,
) -> None:
    config = _asset_fixture(tmp_path)

    report = run_asset_preflight(config)

    assert report.ok
    assert len(report.verified_files) == 4


def test_preflight_detects_changed_core_wheel(tmp_path: Path) -> None:
    config = _asset_fixture(tmp_path)
    (config.wheelhouse_dir / "core.whl").write_bytes(b"changed")

    report = run_asset_preflight(config)

    assert not report.ok
    assert any("SHA-256 mismatch" in error for error in report.errors)
    assert not any(path.endswith("core.whl") for path in report.verified_files)


def test_full_preflight_detects_same_size_changed_model(tmp_path: Path) -> None:
    config = _asset_fixture(tmp_path)
    (config.model_dir / "model.safetensors").write_bytes(b"other")

    report = run_asset_preflight(config)

    assert not report.ok
    assert any("model.safetensors" in error for error in report.errors)


def test_fast_preflight_checks_model_presence_and_size_without_hash(
    tmp_path: Path,
) -> None:
    config = _asset_fixture(tmp_path).model_copy(update={"full_verification": False})
    (config.model_dir / "model.safetensors").write_bytes(b"other")

    report = run_asset_preflight(config)

    assert report.ok
