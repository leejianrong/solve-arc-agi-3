"""Typed runtime configuration for the pinned ARC-AGI-3 baseline."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .baseline_manifest import ControlManifest


class FrozenConfig(BaseModel):
    """Strict immutable configuration shared by every runtime component."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Provider(StrEnum):
    LOCAL_VLLM = "local_vllm"
    RUNPOD = "runpod"
    OPENROUTER = "openrouter"
    FAKE = "fake"


class SamplingConfig(FrozenConfig):
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(gt=0)
    seed: int = 0
    thinking: bool = True


class InferenceConfig(FrozenConfig):
    provider: Provider
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    request_timeout_seconds: float = Field(gt=0)
    context_limit: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    sampling: SamplingConfig

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> InferenceConfig:
        if (
            self.provider in {Provider.OPENROUTER, Provider.RUNPOD}
            and self.api_key_env is None
        ):
            raise ValueError(f"{self.provider.value} requires api_key_env")
        if self.max_output_tokens > self.context_limit:
            raise ValueError("max_output_tokens exceeds context_limit")
        return self


class AssetConfig(FrozenConfig):
    manifest_path: Path
    model_dir: Path
    wheelhouse_dir: Path
    full_verification: bool = False
    offline: bool = True


class VllmLaunchConfig(FrozenConfig):
    executable: tuple[str, ...] = (
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
    )
    host: str = "127.0.0.1"
    port: int = Field(ge=1, le=65535)
    served_model_name: str = Field(min_length=1)
    tensor_parallel_size: int = Field(gt=0)
    max_model_len: int = Field(gt=0)
    tool_call_parser: str = Field(min_length=1)
    reasoning_parser: str = Field(min_length=1)
    generation_config: str = Field(min_length=1)
    enable_prefix_caching: bool = True
    preserve_thinking: bool = True
    startup_timeout_seconds: float = Field(default=900, gt=0)
    shutdown_timeout_seconds: float = Field(default=30, gt=0)
    readiness_poll_seconds: float = Field(default=1, gt=0)


class BaselineConfig(FrozenConfig):
    inference: InferenceConfig
    assets: AssetConfig
    vllm: VllmLaunchConfig

    @model_validator(mode="after")
    def validate_limits(self) -> BaselineConfig:
        if self.inference.context_limit > self.vllm.max_model_len:
            raise ValueError("inference context_limit exceeds vLLM max_model_len")
        if self.inference.max_output_tokens > self.inference.context_limit:
            raise ValueError("max_output_tokens exceeds inference context_limit")
        return self


def config_from_manifest(
    manifest: ControlManifest,
    *,
    manifest_path: Path,
    model_dir: Path,
    wheelhouse_dir: Path,
    provider: Provider = Provider.LOCAL_VLLM,
    base_url: str | None = None,
    model: str | None = None,
    api_key_env: str | None = None,
    max_output_tokens: int = 4096,
    full_verification: bool = False,
    offline: bool = True,
) -> BaselineConfig:
    """Create runtime configuration without changing any pinned control setting."""

    serving = manifest.serving
    generation = manifest.generation
    endpoint = base_url or f"http://{serving.host}:{serving.port}/v1"
    return BaselineConfig(
        inference=InferenceConfig(
            provider=provider,
            base_url=endpoint,
            model=model or serving.served_model_name,
            api_key_env=api_key_env,
            request_timeout_seconds=generation.request_timeout_seconds,
            context_limit=serving.agent_context_window,
            max_output_tokens=generation.max_output_tokens or max_output_tokens,
            sampling=SamplingConfig(
                temperature=generation.temperature,
                top_p=generation.top_p,
                top_k=generation.top_k,
                thinking=generation.thinking,
            ),
        ),
        assets=AssetConfig(
            manifest_path=manifest_path,
            model_dir=model_dir,
            wheelhouse_dir=wheelhouse_dir,
            full_verification=full_verification,
            offline=offline,
        ),
        vllm=VllmLaunchConfig(
            host=serving.host,
            port=serving.port,
            served_model_name=serving.served_model_name,
            tensor_parallel_size=serving.tensor_parallel_size,
            max_model_len=serving.max_model_len,
            tool_call_parser=serving.tool_call_parser,
            reasoning_parser=serving.reasoning_parser,
            generation_config=serving.generation_config,
            enable_prefix_caching=serving.enable_prefix_caching,
            preserve_thinking=serving.preserve_thinking,
        ),
    )
