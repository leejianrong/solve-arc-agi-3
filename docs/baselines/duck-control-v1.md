# Duck control v1 provenance and clean-room boundary

Date captured: 2026-09-20

Machine-readable pin: [`../../configs/baselines/duck-control-v1.json`](../../configs/baselines/duck-control-v1.json)

## Decision

Use Duck as a behavioural control, not as vendored source.

The inspected upstream revision is `7652836056c59e044f093e3c13ed7438c814169e`. Its `ARC3-Inference/pyproject.toml` contains the classifier `License :: OSI Approved :: MIT License`, but the repository has no `LICENSE`, `COPYING`, or `NOTICE` file. GitHub's repository API also reports no detected licence. A package classifier is useful evidence of intent, but it is not the concrete licence text needed to copy and redistribute the implementation.

Until Tufa Labs adds a licence file or provides another explicit grant, this project will not copy Duck source, prompts, tests, or notebook cells. We will implement the public interface and documented behaviour with original code and original prompt wording. The manifest records hashes of inspected upstream files so later reviews can identify exactly which version informed the specification.

## Pinned inputs

### Harness reference

- Repository: `https://github.com/Tufalabs/duck-harness`
- Commit: `7652836056c59e044f093e3c13ed7438c814169e`
- Commit time: `2026-07-01T16:59:32+02:00`
- Reuse policy: clean-room behavioural reimplementation only

### Model

- Hugging Face repository: `vrfai/Qwen3.6-27B-FP8`
- Revision used by the Kaggle snapshot: `076636763143ca2b7cf0d66e16a09c2a0a689dfa`
- Licence: Apache-2.0
- Kaggle dataset: `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`, dataset ID `10453841`, version 1
- `model.safetensors`: 35,923,194,376 bytes, SHA-256 `00123c8e4bc39fb738512545f59e17c8c568c607905d9f812bb9adb884400001`
- `tokenizer.json`: 19,989,424 bytes, SHA-256 `e56427d66f44411c2dec1288b236f6d2c3eeafd611d1d0e2e92ad9301616e1e7`

The Kaggle dataset contains `hf-provenance.json`, which names this older revision. Pinning the current Hugging Face head would silently change the control.

### Runtime wheelhouse

- Kaggle dataset: `driessmit1/arc3-vllm-h100-wheelhouse-v3`, dataset ID `10283361`, version 1
- Published checksum entries: 179
- `SHA256SUMS`: 21,285 bytes, SHA-256 `44029b360a9c0073e4b0add10703fc3386dc06bf1314f5913a0dad9564144cbb`
- Core runtime: Python 3.12.12, vLLM 0.19.0, Torch 2.10.0, FlashInfer 0.6.6, Transformers 4.57.6, and Triton 3.6.0

The machine-readable manifest pins the checksum for every core wheel and the upstream `requirements.lock`. The source dataset also contains the full checksum list for all 179 files. Before packaging, the artifact builder must verify the checksum-list hash, the expected core entries, and the actual downloaded files.

The Kaggle dataset declares `Other (specified in description)` while its public description is empty. We may attach it as a public Kaggle input for the control run, but we will not republish its assembled contents. A submission artifact we own should carry the individual dependency licence notices and a wheelhouse produced from our own lock.

## Behavioural specification

The clean-room control will reproduce these observable properties without copying implementation text.

### Observation and state

- The current 64 by 64 board is available as an upscaled image and a letter-coded ASCII grid.
- A deterministic 4-connected segmentation provides colour, shape hash, size, boundary, containment, and adjacency.
- The Python tool receives the current frame, previous frame, episode history, explicit before and after transitions, legal actions, and the latest action result.
- A compact episode-local summary retains the world model, goal model, action model, recent findings, open questions, current plan, and cross-level notes.
- No state transfers between independent environments.

### Tool and actions

- The model sees one function named `python` with one required string argument named `code`.
- Each Python snippet is ephemeral and runs for at most 30 seconds in a restricted subprocess.
- The snippet can inspect structured state and call `action(actions)` one or more times.
- Engine actions map to `UP`, `DOWN`, `LEFT`, `RIGHT`, `SPACE`, `MOUSE`, and `RESET`. Mouse actions use integer row and column coordinates.
- Tool output is capped at 1,024 tokens.

### Model call

- Thinking is enabled.
- Temperature is 0.6, top-p is 0.95, and top-k is 20.
- The agent context window is 32,768 tokens.
- The server can expose up to 65,536 model tokens.
- The model may choose the Python tool automatically and may make multiple tool calls before yielding.
- The prompt asks the model to maintain a compact world model, use segmentation before scanning raw ASCII, test uncertain mechanics, search when the objective is understood, and verify changes after actions. Our prompt will express these requirements in new wording.

### vLLM server

- OpenAI-compatible endpoint at `127.0.0.1:1234/v1`
- One GPU and tensor parallel size 1
- Served name `vrfai/Qwen3.6-27B-FP8`
- Tool parser `qwen3_coder`
- Reasoning parser `qwen3`
- vLLM generation config
- Prefix caching enabled
- Thinking preserved in the chat template

## Verification commands

Load the manifest and confirm its schema:

```sh
uv run python -c 'from pathlib import Path; from solve_arc_agi_3.baseline_manifest import load_control_manifest; print(load_control_manifest(Path("configs/baselines/duck-control-v1.json")).control_id)'
```

After attaching the Kaggle datasets, verify the model files with `verify_pinned_files`. Validate the downloaded `SHA256SUMS` text with `validate_wheelhouse_checksum_manifest`, then verify the wheel files against the checksums before installation.

## Sources

- [Duck harness repository](https://github.com/Tufalabs/duck-harness)
- [Duck technical report](https://tufalabs.ai/research/duck-harness/)
- [Pinned Qwen3.6 27B FP8 revision](https://huggingface.co/vrfai/Qwen3.6-27B-FP8/tree/076636763143ca2b7cf0d66e16a09c2a0a689dfa)
- [Kaggle vLLM wheelhouse](https://www.kaggle.com/datasets/driessmit1/arc3-vllm-h100-wheelhouse-v3)
- [Kaggle Qwen model snapshot](https://www.kaggle.com/datasets/driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot)
