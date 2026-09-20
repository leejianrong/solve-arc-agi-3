"""Typed provenance manifest and checksum helpers for baseline controls."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

Sha256 = str


class FrozenModel(BaseModel):
    """Reject undeclared manifest fields and make loaded pins immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FilePin(FrozenModel):
    name: str = Field(min_length=1)
    sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int | None = Field(default=None, gt=0)


class KaggleDatasetPin(FrozenModel):
    ref: str = Field(pattern=r"^[^/]+/[^/]+$")
    id: int = Field(gt=0)
    version: int = Field(gt=0)
    last_updated: str = Field(min_length=1)
    total_bytes: int = Field(gt=0)


class UpstreamPin(FrozenModel):
    repository: str = Field(pattern=r"^https://")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    committed_at: str = Field(min_length=1)
    license_claim: str = Field(min_length=1)
    license_file_present: bool
    github_detected_license: str | None
    reuse_policy: str = Field(min_length=1)
    source_files: tuple[FilePin, ...] = Field(min_length=1)


class ModelPin(FrozenModel):
    repository: str = Field(pattern=r"^[^/]+/[^/]+$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: str = Field(min_length=1)
    kaggle_dataset: KaggleDatasetPin
    files: tuple[FilePin, ...] = Field(min_length=1)


class WheelhousePin(FrozenModel):
    kaggle_dataset: KaggleDatasetPin
    checksum_manifest: FilePin
    checksum_manifest_url: str = Field(pattern=r"^https://")
    checksum_entry_count: int = Field(gt=0)
    packages: tuple[FilePin, ...] = Field(min_length=1)


class RuntimePin(FrozenModel):
    python: str
    vllm: str
    torch: str
    flashinfer: str
    transformers: str
    triton: str


class ServingPin(FrozenModel):
    served_model_name: str
    host: str
    port: int = Field(ge=1, le=65535)
    tensor_parallel_size: int = Field(gt=0)
    max_model_len: int = Field(gt=0)
    agent_context_window: int = Field(gt=0)
    tool_call_parser: str
    reasoning_parser: str
    generation_config: str
    enable_prefix_caching: bool
    preserve_thinking: bool


class GenerationPin(FrozenModel):
    thinking: bool
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    request_timeout_seconds: int = Field(gt=0)
    tool_timeout_seconds: int = Field(gt=0)
    tool_output_tokens: int = Field(gt=0)


class AgentContractPin(FrozenModel):
    observation: str
    history: str
    tool: str
    action_map: dict[str, str]
    python_modules: tuple[str, ...]
    persistent_summary_fields: tuple[str, ...]
    persistent_assistant_turns: int = Field(ge=0)


class ControlManifest(FrozenModel):
    schema_version: int = Field(ge=1)
    control_id: str = Field(min_length=1)
    captured_at: str = Field(min_length=1)
    upstream: UpstreamPin
    model: ModelPin
    wheelhouse: WheelhousePin
    runtime: RuntimePin
    serving: ServingPin
    generation: GenerationPin
    agent_contract: AgentContractPin


def load_control_manifest(path: Path) -> ControlManifest:
    """Load and strictly validate a control manifest."""

    return ControlManifest.model_validate_json(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading a model or wheel into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256sums(text: str) -> dict[str, str]:
    """Parse the two-column SHA256SUMS format used by the pinned wheelhouse."""

    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid SHA256SUMS line {line_number}: {raw_line!r}")
        digest, name = parts
        normalized_name = name.lstrip("*")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"invalid SHA-256 on line {line_number}: {digest!r}")
        if not normalized_name or normalized_name in entries:
            raise ValueError(
                f"missing or duplicate filename on line {line_number}: {name!r}"
            )
        entries[normalized_name] = digest
    return entries


def validate_wheelhouse_checksum_manifest(
    manifest: ControlManifest, checksum_text: str
) -> list[str]:
    """Return reproducibility errors for a downloaded wheelhouse checksum file."""

    errors: list[str] = []
    actual_manifest_hash = hashlib.sha256(checksum_text.encode("utf-8")).hexdigest()
    expected_manifest_hash = manifest.wheelhouse.checksum_manifest.sha256
    if actual_manifest_hash != expected_manifest_hash:
        errors.append(
            "wheelhouse checksum manifest hash mismatch: "
            f"expected {expected_manifest_hash}, got {actual_manifest_hash}"
        )

    entries = parse_sha256sums(checksum_text)
    if len(entries) != manifest.wheelhouse.checksum_entry_count:
        errors.append(
            "wheelhouse checksum entry count mismatch: "
            f"expected {manifest.wheelhouse.checksum_entry_count}, got {len(entries)}"
        )
    for package in manifest.wheelhouse.packages:
        actual_hash = entries.get(package.name)
        if actual_hash is None:
            errors.append(f"wheelhouse checksum is missing {package.name}")
        elif actual_hash != package.sha256:
            errors.append(
                f"wheelhouse checksum mismatch for {package.name}: "
                f"expected {package.sha256}, got {actual_hash}"
            )
    return errors


def verify_pinned_files(root: Path, pins: tuple[FilePin, ...]) -> list[str]:
    """Verify pinned files below a model, wheelhouse, or upstream checkout."""

    errors: list[str] = []
    for pin in pins:
        path = root / pin.name
        if not path.is_file():
            errors.append(f"missing pinned file: {path}")
            continue
        if pin.size_bytes is not None and path.stat().st_size != pin.size_bytes:
            errors.append(
                f"size mismatch for {path}: expected {pin.size_bytes}, "
                f"got {path.stat().st_size}"
            )
            continue
        actual_hash = sha256_file(path)
        if actual_hash != pin.sha256:
            errors.append(
                f"SHA-256 mismatch for {path}: expected {pin.sha256}, got {actual_hash}"
            )
    return errors
