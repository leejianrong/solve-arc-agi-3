import json
from pathlib import Path

import pytest

from solve_arc_agi_3.agent import (
    BaselineAgentCore,
    OfficialAgentAdapter,
    parse_action_decision,
)
from solve_arc_agi_3.baseline_config import Provider, config_from_manifest
from solve_arc_agi_3.baseline_manifest import load_control_manifest
from solve_arc_agi_3.fake_openai import DeterministicFakeServer
from solve_arc_agi_3.inference import (
    FunctionCall,
    InferenceResponse,
    OpenAICompatibleClient,
    ToolCall,
)
from solve_arc_agi_3.smoke import (
    SmokeFrame,
    _run_smoke_episode,
    run_deterministic_smoke,
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


def _read_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_deterministic_smoke_exercises_complete_no_network_path(
    tmp_path: Path,
) -> None:
    result = run_deterministic_smoke(
        config=_fake_config(tmp_path),
        runs_root=tmp_path / "runs",
        run_id_factory=lambda: "run-one",
    )

    events = _read_trace(result.trace_path)
    assert result.terminal_state == "WIN"
    assert result.actions == ("ACTION1",)
    assert result.model_requests == 1
    assert result.network_requests == 0
    assert [event["event"] for event in events] == [
        "configuration",
        "observation",
        "model_request",
        "model_response",
        "action_decision",
        "environment_transition",
        "terminal",
    ]
    assert events[3]["data"]["usage"]["total_tokens"] == 40
    content = events[2]["data"]["messages"][-1]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_two_smoke_runs_have_isolated_files_prompts_and_episode_state(
    tmp_path: Path,
) -> None:
    config = _fake_config(tmp_path)
    ids = iter(["run-one", "run-two"])
    first = run_deterministic_smoke(
        config=config,
        runs_root=tmp_path / "runs",
        run_id_factory=lambda: next(ids),
    )
    second = run_deterministic_smoke(
        config=config,
        runs_root=tmp_path / "runs",
        run_id_factory=lambda: next(ids),
    )

    assert first.run_dir != second.run_dir
    assert first.trace_path != second.trace_path
    assert "run-one" not in second.trace_path.read_text()
    assert (
        _read_trace(first.trace_path)[1]["data"]
        == _read_trace(second.trace_path)[1]["data"]
    )
    assert first.actions == second.actions == ("ACTION1",)


def test_official_adapter_uses_choose_action_and_is_done_contract(
    tmp_path: Path,
) -> None:
    config = _fake_config(tmp_path)
    server = DeterministicFakeServer(config.model)
    core = BaselineAgentCore(OpenAICompatibleClient(config, transport=server))
    adapter = OfficialAgentAdapter(core, action_mapper=lambda decision: decision.name)
    frame = SmokeFrame(
        game_id="game",
        frame=[[[1]]],
        state="NOT_FINISHED",
        levels_completed=0,
        win_levels=1,
        available_actions=[1],
    )

    assert adapter.choose_action([], frame) == "ACTION1"
    assert adapter.is_done([], frame) is False
    terminal = SmokeFrame(
        game_id="game",
        frame=[],
        state="WIN",
        levels_completed=1,
        win_levels=1,
        available_actions=[],
    )
    assert adapter.is_done([frame], terminal) is True


def test_python_tool_parses_official_mouse_coordinates() -> None:
    response = InferenceResponse(
        tool_calls=(
            ToolCall(
                id="mouse",
                function=FunctionCall(
                    name="python",
                    arguments=json.dumps(
                        {
                            "code": (
                                'action([{"action": "ACTION6", "row": 12, '
                                '"column": 34}])'
                            )
                        }
                    ),
                ),
            ),
        ),
        elapsed_seconds=0,
    )

    decision = parse_action_decision(response, available_actions=[6])

    assert decision.name == "ACTION6"
    assert decision.data == {"x": 34, "y": 12}


class _UnparseableClient:
    """A model client whose reply never names an available ARC action."""

    def complete(self, request: object) -> InferenceResponse:
        del request
        return InferenceResponse(content="I am thinking it over.", elapsed_seconds=0)


def test_a_failed_parse_still_traces_the_request_and_response(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="did not contain an available"):
        _run_smoke_episode(
            config=_fake_config(tmp_path),
            runs_root=tmp_path / "runs",
            client=_UnparseableClient(),
            request_capture=None,
            network_requests=1,
            run_id_factory=lambda: "run-unparseable",
        )

    events = _read_trace(tmp_path / "runs" / "run-unparseable" / "trace.jsonl")
    assert [event["event"] for event in events] == [
        "configuration",
        "observation",
        "model_request",
        "model_response",
        "error",
    ]
    assert events[3]["data"]["content"] == "I am thinking it over."


def test_direct_text_fallback_uses_final_available_action_with_coordinates() -> None:
    response = InferenceResponse(
        content="I considered ACTION1, but the target is ACTION6(34, 12)",
        elapsed_seconds=0,
    )

    decision = parse_action_decision(response, available_actions=[1, 6])

    assert decision.name == "ACTION6"
    assert decision.data == {"x": 34, "y": 12}
