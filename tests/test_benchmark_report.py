from solve_arc_agi_3.benchmark_report import parse_scorecards_from_log, summarize

FIXTURE_LOG = """\
2026-08-31 10:00:00 | INFO | Game list: ['ls20-abc123']
2026-08-31 10:05:00 | INFO | Received SIGINT, exiting...
2026-08-31 10:05:00 | INFO | --- EXISTING SCORECARD REPORT ---
2026-08-31 10:05:00 | INFO | {
  "card_id": "card-1",
  "cards": [
    {
      "game_id": "ls20-abc123",
      "score": 0.42,
      "actions": 1337
    }
  ]
}
2026-08-31 10:05:01 | INFO | View your scorecard online: https://arcprize.org/scorecards/card-1
"""

FIXTURE_LOG_SINGLE_GAME = """\
2026-08-31 10:00:00 | INFO | --- EXISTING SCORECARD REPORT ---
2026-08-31 10:00:00 | INFO | {
  "game_id": "ls20-xyz789",
  "rhae": 0.77,
  "action_count": 42
}
"""


def test_parse_scorecards_from_log_finds_json_block() -> None:
    scorecards = parse_scorecards_from_log(FIXTURE_LOG)
    assert len(scorecards) == 1
    assert scorecards[0]["card_id"] == "card-1"


def test_parse_scorecards_from_log_returns_empty_list_when_no_marker() -> None:
    assert parse_scorecards_from_log("nothing to see here") == []


def test_summarize_extracts_rhae_and_actions_from_cards_list() -> None:
    report = summarize(log_text=FIXTURE_LOG, tokens_per_second=12.5)
    assert len(report.games) == 1
    game = report.games[0]
    assert game.game_id == "ls20-abc123"
    assert game.rhae == 0.42
    assert game.actions == 1337
    assert report.tokens_per_second == 12.5
    assert report.mean_rhae == 0.42


def test_summarize_handles_single_game_top_level_scorecard() -> None:
    report = summarize(log_text=FIXTURE_LOG_SINGLE_GAME)
    assert len(report.games) == 1
    game = report.games[0]
    assert game.game_id == "ls20-xyz789"
    assert game.rhae == 0.77
    assert game.actions == 42


def test_mean_rhae_is_none_when_no_games_scored() -> None:
    report = summarize(log_text="no scorecards here")
    assert report.games == []
    assert report.mean_rhae is None
