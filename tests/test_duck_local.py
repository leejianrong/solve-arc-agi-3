import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "duck_local",
    Path(__file__).resolve().parents[1] / "scripts" / "duck_local" / "duck_local.py",
)
assert _spec and _spec.loader
duck_local = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(duck_local)


def _nb(tmp_path: Path, cells: list[dict]) -> Path:
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps({"cells": cells}))
    return p


def test_extract_patch_drops_magic_line(tmp_path: Path) -> None:
    cells = [
        {"cell_type": "markdown", "source": ["# hi"]},
        {
            "cell_type": "code",
            "source": [
                "%%writefile /kaggle/harness-changes.patch\n",
                "diff --git a/x b/x\n",
                "+y",
            ],
        },
    ]
    assert duck_local.extract_patch(_nb(tmp_path, cells)) == "diff --git a/x b/x\n+y\n"


def test_extract_patch_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        duck_local.extract_patch(
            _nb(tmp_path, [{"cell_type": "code", "source": ["print(1)"]}])
        )


def test_env_is_text_only() -> None:
    assert duck_local.HARNESS_ENV["MULTIMODAL_CONTEXT"] != "current_grid"
    assert duck_local.HARNESS_ENV["ARC3_DIFF_IMAGE"] == "0"
