import shutil
from pathlib import Path

import jsonschema
import pandas
import pytest

import asreview as asr
from asreview.project.migration import detect_version
from asreview.project.schema import SCHEMA


@pytest.fixture
def asreview_v2_project(tmpdir):
    """Fixture to set up a test project for ASReview."""
    test_state_fp = Path("tests", "asreview_files", "asreview-demo-project-v2.asreview")
    tmp_project_path = Path(tmpdir, "asreview-demo-project-v2.asreview")
    shutil.copy(test_state_fp, tmp_project_path)
    return tmp_project_path


def assert_valid_project(project):
    assert detect_version(project.config) == 4
    jsonschema.validate(instance=project.config, schema=SCHEMA)

    with project.db as db:
        db.get_results_table()
        db.get_last_ranking_table()
        db.get_decision_changes()
        assert isinstance(db.input["title"], pandas.Series)
        assert isinstance(db.input["included"], pandas.Series)

        # Verify v4 tables exist
        cur = db._conn.cursor()
        tables = [
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]
        assert "tag_groups" in tables
        assert "tag_options" in tables
        assert "list_containers" in tables

        # Verify tag_groups uses group_id (UUID v7 TEXT PK)
        cols = {row[1] for row in cur.execute("PRAGMA table_info(tag_groups)")}
        assert "group_id" in cols
        assert "input_helper_text" in cols
        assert "tag_values" not in cols
        assert "description" not in cols

        # Verify tag_options exists with expected columns
        opt_cols = {row[1] for row in cur.execute("PRAGMA table_info(tag_options)")}
        assert "option_id" in opt_cols
        assert "group_id" in opt_cols
        assert "free_text_enabled" in opt_cols
        assert "free_text_required" in opt_cols

        # Verify tags uses tag_id (UUID v7 TEXT PK) and option_id
        tag_cols = {row[1] for row in cur.execute("PRAGMA table_info(tags)")}
        assert "tag_id" in tag_cols
        assert "option_id" in tag_cols

    cycle_data = asr.ActiveLearningCycleData(**project.get_model_config())
    cycle = asr.ActiveLearningCycle.from_meta(cycle_data)

    assert isinstance(cycle.classifier.name, str)


def test_project_migration_1_to_3(tmpdir):
    asreview_v1_file = Path(
        "asreview",
        "webapp",
        "tests",
        "asreview-project-file-archive",
        "v1.5",
        "asreview-project-v1-5-startreview.asreview",
    )
    assert asreview_v1_file.exists()
    project = asr.Project.load(open(asreview_v1_file, "rb"), tmpdir, safe_import=True)
    assert_valid_project(project)


def test_project_migration_2_to_3(tmpdir, asreview_v2_project):
    project = asr.Project.load(
        open(asreview_v2_project, "rb"), tmpdir, safe_import=True
    )
    assert_valid_project(project)
