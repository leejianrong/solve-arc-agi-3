from pathlib import Path

from solve_arc_agi_3.baseline_manifest import (
    FilePin,
    load_control_manifest,
    parse_sha256sums,
    validate_wheelhouse_checksum_manifest,
    verify_pinned_files,
)

PROJECT_ROOT = Path(__file__).parents[1]
MANIFEST_PATH = PROJECT_ROOT / "configs/baselines/duck-control-v1.json"


def test_duck_control_manifest_loads_with_immutable_pins() -> None:
    manifest = load_control_manifest(MANIFEST_PATH)

    assert manifest.control_id == "duck-clean-room-v1"
    assert manifest.upstream.revision == ("7652836056c59e044f093e3c13ed7438c814169e")
    assert manifest.upstream.license_file_present is False
    assert manifest.upstream.github_detected_license is None
    assert manifest.upstream.reuse_policy == (
        "clean-room-behavioral-reimplementation-only"
    )
    assert manifest.model.revision == "076636763143ca2b7cf0d66e16a09c2a0a689dfa"
    assert manifest.model.license == "Apache-2.0"
    assert manifest.runtime.vllm == "0.19.0"
    assert manifest.serving.max_model_len == 65536
    assert manifest.serving.agent_context_window == 32768


def test_parse_sha256sums_accepts_binary_marker() -> None:
    digest_a = "a" * 64
    digest_b = "b" * 64

    assert parse_sha256sums(f"{digest_a}  alpha.whl\n{digest_b} *beta.whl\n") == {
        "alpha.whl": digest_a,
        "beta.whl": digest_b,
    }


def test_checksum_manifest_validation_reports_changed_package() -> None:
    manifest = load_control_manifest(MANIFEST_PATH)
    lines = [
        f"{package.sha256}  {package.name}" for package in manifest.wheelhouse.packages
    ]
    lines.extend(
        f"{'f' * 64}  filler-{index}.whl"
        for index in range(manifest.wheelhouse.checksum_entry_count - len(lines))
    )
    text = "\n".join(lines) + "\n"

    errors = validate_wheelhouse_checksum_manifest(manifest, text)

    assert len(errors) == 1
    assert errors[0].startswith("wheelhouse checksum manifest hash mismatch")


def test_verify_pinned_files_checks_size_and_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"baseline")
    good_pin = FilePin(
        name="artifact.bin",
        sha256="8ba8496a2525ae171ffd104d632dede6ef418d9b95962a9d88e2fcdbc8d48d24",
        size_bytes=8,
    )

    assert verify_pinned_files(tmp_path, (good_pin,)) == []

    bad_pin = FilePin(name="artifact.bin", sha256="0" * 64, size_bytes=8)
    errors = verify_pinned_files(tmp_path, (bad_pin,))
    assert len(errors) == 1
    assert errors[0].startswith("SHA-256 mismatch")
