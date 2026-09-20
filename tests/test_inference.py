from pathlib import Path

import pytest

from solve_arc_agi_3.baseline_config import Provider, config_from_manifest
from solve_arc_agi_3.baseline_manifest import load_control_manifest
from solve_arc_agi_3.fake_openai import DeterministicFakeServer
from solve_arc_agi_3.inference import (
    ChatMessage,
    InferenceRequest,
    ModelsProbe,
    OpenAICompatibleClient,
    ToolDefinition,
)

PROJECT_ROOT = Path(__file__).parents[1]
MANIFEST_PATH = PROJECT_ROOT / "configs/baselines/duck-control-v1.json"


def _fake_config(tmp_path: Path):
    manifest = load_control_manifest(MANIFEST_PATH)
    return config_from_manifest(
        manifest,
        manifest_path=MANIFEST_PATH,
        model_dir=tmp_path / "model",
        wheelhouse_dir=tmp_path / "wheels",
        provider=Provider.FAKE,
        base_url="http://fake.invalid/v1",
    ).inference


def test_openai_client_preserves_reasoning_tools_usage_and_sampling(
    tmp_path: Path,
) -> None:
    config = _fake_config(tmp_path)
    server = DeterministicFakeServer(config.model)
    ticks = iter([10.0, 10.25])
    client = OpenAICompatibleClient(
        config, transport=server, monotonic=lambda: next(ticks)
    )
    request = InferenceRequest(
        messages=(ChatMessage(role="user", content="observe"),),
        tools=(
            ToolDefinition(
                name="python",
                description="choose",
                parameters={"type": "object"},
            ),
        ),
    )

    response = client.complete(request)

    assert response.reasoning == "The first legal move reaches the goal."
    assert response.tool_calls[0].function.name == "python"
    assert response.usage.total_tokens == 40
    assert response.elapsed_seconds == 0.25
    assert server.requests[0]["model"] == config.model
    assert server.requests[0]["temperature"] == 0.6
    assert server.requests[0]["top_k"] == 20
    assert server.network_requests == 0


def test_models_probe_uses_same_fake_server_without_network(tmp_path: Path) -> None:
    config = _fake_config(tmp_path)
    server = DeterministicFakeServer(config.model)
    probe = ModelsProbe(
        base_url=config.base_url,
        timeout_seconds=1,
        transport=server,
    )

    assert probe.is_ready(config.model) is True
    assert probe.is_ready("different-model") is False
    assert server.network_requests == 0


def test_client_requires_remote_api_key_at_call_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = load_control_manifest(MANIFEST_PATH)
    config = config_from_manifest(
        manifest,
        manifest_path=MANIFEST_PATH,
        model_dir=tmp_path / "model",
        wheelhouse_dir=tmp_path / "wheels",
        provider=Provider.OPENROUTER,
        base_url="https://openrouter.ai/api/v1",
        model="provider/model-id",
        api_key_env="TEST_OPENROUTER_API_KEY",
    ).inference
    monkeypatch.delenv("TEST_OPENROUTER_API_KEY", raising=False)
    client = OpenAICompatibleClient(
        config, transport=DeterministicFakeServer(config.model)
    )

    with pytest.raises(ValueError, match="TEST_OPENROUTER_API_KEY"):
        client.complete(
            InferenceRequest(messages=(ChatMessage(role="user", content="observe"),))
        )


def test_openrouter_request_explicitly_enables_reasoning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = load_control_manifest(MANIFEST_PATH)
    config = config_from_manifest(
        manifest,
        manifest_path=MANIFEST_PATH,
        model_dir=tmp_path / "model",
        wheelhouse_dir=tmp_path / "wheels",
        provider=Provider.OPENROUTER,
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3.6-27b",
        api_key_env="TEST_OPENROUTER_API_KEY",
    ).inference
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "test-only-placeholder")
    server = DeterministicFakeServer(config.model)

    OpenAICompatibleClient(config, transport=server).complete(
        InferenceRequest(messages=(ChatMessage(role="user", content="observe"),))
    )

    assert server.requests[0]["reasoning"] == {"enabled": True}
