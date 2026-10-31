# Testing the agent with semula and Haiku subagents

A model-in-the-loop check with no hosted API: the agent calls semula, an OpenAI-compatible server that
parks every unknown prompt until a coding agent answers it. Haiku subagents play the part of the model.
It tests whether prompts and observations are understandable, not how Qwen behaves, and 50 calls per game
is too few to measure level clears.

## Start the server

```sh
cd ~/projects/abang-ai/semula
curl -s 127.0.0.1:8080/health || SEMULA_MODE=record docker compose -p semula-arc3 up -d --wait semula
```

Dashboard: http://127.0.0.1:8080/ui/ (every prompt and answer). Stop it with
`docker compose -p semula-arc3 down`. Semula replays identical prompts from its store, so an old wrong answer
replays too; use a fresh project or volume for a clean run.

## Run the game side

```sh
set -a; . ./.env; set +a   # not needed for semula, only for other tools
PYTHONPATH=src ../play-arc-agi-3/.venv/bin/python scripts/local_play.py \
  --games ls20,ft09,cd82 --steps 60 --config configs/baselines/semula-haiku.dev.json --timeout 600 --concurrency 3
```

## The players

Spawn one Haiku subagent per game (`model: haiku`, background, within the 2-subagent cap, so run two at a
time and the third after one ends). Shard by the `Available actions:` header, which sits in the first 200
characters of each call (`semula pending` truncates `last_message` there). Each player loops, over Bash only:

```sh
docker exec semula-arc3-semula-1 semula pending --wait 60 --json
docker exec semula-arc3-semula-1 semula show <ID> --json
docker exec -i semula-arc3-semula-1 semula answer <ID> --stdin <<'EOF'
{"message": {"content": null, "tool_calls": [{"function": {"name": "python", "arguments": {"code": "action([1, 4])"}}}]}}
EOF
```

Rules for the players: use only the request contents (no repo or game source), answer only their game, stop
after 4 empty waits or 50 answers, then report specifically what they could not tell from the text.

## What it cost and gave

Three players answered about 135 calls in about 5 minutes at 95 to 105k tokens each. The reports are claims:
check them against our own logs. They found real bugs in our observation text each time.
