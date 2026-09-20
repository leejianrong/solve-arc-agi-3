from pathlib import Path

import pytest
from pydantic import ValidationError

from solve_arc_agi_3.baseline_config import (
    InferenceConfig,
    Provider,
    config_from_manifest,
)
from solve_arc_agi_3.baseline_manifest import load_control_manifest

PROJECT_ROOT = Path(__file__).parents[1]
MANIFEST_PATH = PROJECT_ROOT / "configs/baselines/duck-control-v1.json"
OPENROUTER_CONFIG_PATH = (
    PROJECT_ROOT / "configs/baselines/openrouter-qwen36-27b.dev.json"
)


def test_config_from_manifest_preserves_control_pins(tmp_path: Path) -> None:
    manifest = load_control_manifest(MANIFEST_PATH)

    config = config_from_manifest(
        manifest,
        manifest_path=MANIFEST_PATH,
        model_dir=tmp_path / "model",
        wheelhouse_dir=tmp_path / "wheels",
    )

    assert config.inference.provider is Provider.LOCAL_VLLM
    assert config.inference.base_url == "http://127.0.0.1:1234/v1"
    assert config.inference.model == "vrfai/Qwen3.6-27B-FP8"
    assert config.inference.api_key_env is None
    assert config.inference.context_limit == 32768
    assert config.vllm.max_model_len == 65536
    assert config.vllm.tool_call_parser == "qwen3_coder"
    assert config.vllm.reasoning_parser == "qwen3"


def test_remote_provider_requires_api_key_environment_name(tmp_path: Path) -> None:
    manifest = load_control_manifest(MANIFEST_PATH)

    with pytest.raises(ValidationError, match="requires api_key_env"):
        config_from_manifest(
            manifest,
            manifest_path=MANIFEST_PATH,
            model_dir=tmp_path / "model",
            wheelhouse_dir=tmp_path / "wheels",
            provider=Provider.OPENROUTER,
            base_url="https://openrouter.ai/api/v1",
            model="provider/model-id",
        )


def test_remote_config_stores_only_environment_variable_name(tmp_path: Path) -> None:
    manifest = load_control_manifest(MANIFEST_PATH)
    config = config_from_manifest(
        manifest,
        manifest_path=MANIFEST_PATH,
        model_dir=tmp_path / "model",
        wheelhouse_dir=tmp_path / "wheels",
        provider=Provider.RUNPOD,
        base_url="https://example.invalid/v1",
        api_key_env="RUNPOD_API_KEY",
    )

    serialized = config.model_dump_json()
    assert config.inference.api_key_env == "RUNPOD_API_KEY"
    assert "Bearer" not in serialized


def test_checked_openrouter_development_config_is_typed_and_unsecret() -> None:
    config = InferenceConfig.model_validate_json(
        OPENROUTER_CONFIG_PATH.read_text(encoding="utf-8")
    )

    assert config.provider is Provider.OPENROUTER
    assert config.model == "qwen/qwen3.6-27b"
    assert config.api_key_env == "OPENROUTER_API_KEY"
    assert "Bearer" not in config.model_dump_json()
