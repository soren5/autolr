import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.smoke
def test_race_behavior_key_analysis_runs_on_existing_facilitated_mutation_dump(tmp_path):
    dump_dir = Path("dumps/facilitated_mutation_base/run_1")
    if not dump_dir.is_dir():
        pytest.skip("facilitated_mutation_base dump is not available")

    watched_file = dump_dir / "iteration_0.json"
    before_mtime = watched_file.stat().st_mtime_ns
    result = subprocess.run(
        [
            sys.executable,
            "utils/analyze_race_behavior_keys.py",
            "--root",
            str(dump_dir),
            "--output-dir",
            str(tmp_path),
            "--examples",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Race behavior key analysis" in result.stdout
    assert (tmp_path / "_race_behavior_key_summary.csv").is_file()
    groups_path = tmp_path / "_race_behavior_key_groups.json"
    assert groups_path.is_file()
    with groups_path.open("r") as groups_file:
        groups = json.load(groups_file)
    assert "summary" in groups
    assert "groups" in groups
    assert watched_file.stat().st_mtime_ns == before_mtime
