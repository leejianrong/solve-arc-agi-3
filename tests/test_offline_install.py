from pathlib import Path

from solve_arc_agi_3.offline_install import build_offline_install_command


def test_offline_install_command_forbids_index_access(tmp_path: Path) -> None:
    command = build_offline_install_command(tmp_path / "wheels")

    assert "--no-index" in command
    assert "--find-links" in command
    assert not any("http://" in part or "https://" in part for part in command)
    assert command[-1].endswith("requirements.lock")
