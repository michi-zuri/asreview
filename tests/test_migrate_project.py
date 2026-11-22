import json
import shutil
import sqlite3
from pathlib import Path

import jsonschema
import pandas
import pytest

import asreview as asr
from asreview.database.fork_migrations import ASMR_PROJECT_VERSION
from asreview.database.fork_migrations import FORK_APPLICATION_ID
from asreview.database.fork_migrations import FORK_USER_VERSION
from asreview.project.migration import detect_version
from asreview.project.migration import is_supported_version
from asreview.project.migration import migrate_project
from asreview.project.migration import needs_migration
from asreview.project.migration.migrate import _migrate_v4_asmr
from asreview.project.schema import SCHEMA


@pytest.fixture
def asreview_v2_project(tmpdir):
    """Fixture to set up a test project for ASReview."""
    test_state_fp = Path("tests", "asreview_files", "asreview-demo-project-v2.asreview")
    tmp_project_path = Path(tmpdir, "asreview-demo-project-v2.asreview")
    shutil.copy(test_state_fp, tmp_project_path)
    return tmp_project_path


def assert_valid_project(project):
    assert detect_version(project.config) == ASMR_PROJECT_VERSION
    jsonschema.validate(instance=project.config, schema=SCHEMA)

    # The migrated database must carry the asmr mark and all ledger rows.
    assert not Path(project.project_path, "tags.json").exists()
    assert not Path(project.project_path, "lists.json").exists()

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
            for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        assert "tag_groups" in tables
        assert "tag_options" in tables
        assert "list_containers" in tables

        # Verify the fork mark and ledger are in place
        assert cur.execute("PRAGMA application_id").fetchone()[0] == FORK_APPLICATION_ID
        assert cur.execute("PRAGMA user_version").fetchone()[0] == FORK_USER_VERSION
        assert {
            row[0] for row in cur.execute("SELECT revision FROM fork_migrations")
        } == {
            "0000_tags_tables",
            "0001_change_tracking",
            "0002_user_version_mark",
        }

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


def test_import_unmarked_project_refused(tmpdir):
    """A project whose results.db lacks the asmr mark cannot be imported."""
    project_dir = Path(tmpdir, "src.asreview")
    project = asr.Project.create(project_dir)

    # Unmark its results.db before exporting.
    with sqlite3.connect(str(project.db_path)) as conn:
        conn.execute("PRAGMA application_id = 0")
        conn.commit()

    export_fp = Path(tmpdir, "src-export.asreview")
    project.export(export_fp)

    projects_dir = Path(tmpdir, "projects")
    projects_dir.mkdir()

    with pytest.raises(ValueError, match="asmr"):
        asr.Project.load(export_fp, projects_dir)

    assert list(projects_dir.iterdir()) == []


def test_safe_import_rekeys_change_tracking(tmpdir):
    """safe_import re-keys project_state, record_versions and change_log."""
    project_dir = Path(tmpdir, "src.asreview")
    project = asr.Project.create(project_dir)

    # Seed change-tracking rows under the source project's id.
    with sqlite3.connect(str(project.db_path)) as conn:
        old_id = conn.execute("SELECT project_id FROM project_state").fetchone()[0]
        conn.execute(
            "INSERT INTO record_versions (project_id, record_id, version) "
            "VALUES (?, 1, 0)",
            (old_id,),
        )
        conn.execute(
            "INSERT INTO change_log "
            "(project_id, version, record_id, source, changed_at) "
            "VALUES (?, 0, 1, 'test', 0.0)",
            (old_id,),
        )
        conn.commit()

    export_fp = Path(tmpdir, "src-export.asreview")
    project.export(export_fp)

    projects_dir = Path(tmpdir, "projects")
    projects_dir.mkdir()
    imported = asr.Project.load(export_fp, projects_dir, safe_import=True)

    new_id = imported.config["id"]
    assert new_id != old_id

    with sqlite3.connect(str(imported.db_path)) as conn:
        assert (
            conn.execute("SELECT project_id FROM project_state").fetchone()[0] == new_id
        )
        assert conn.execute("SELECT project_id FROM record_versions").fetchall() == [
            (new_id,)
        ]
        assert conn.execute("SELECT project_id FROM change_log").fetchall() == [
            (new_id,)
        ]


def test_version_helpers():
    assert needs_migration(1)
    assert needs_migration(2)
    assert needs_migration(3)
    assert needs_migration(4)
    assert not needs_migration(ASMR_PROJECT_VERSION)
    assert not needs_migration(0)
    assert not needs_migration(5)

    assert is_supported_version(1)
    assert is_supported_version(2)
    assert is_supported_version(3)
    assert is_supported_version(4)
    assert is_supported_version(ASMR_PROJECT_VERSION)
    assert not is_supported_version(0)
    assert not is_supported_version(5)


def test_new_project_is_asmr(tmpdir):
    project = asr.Project.create(Path(tmpdir, "new.asreview"))
    assert detect_version(project.config) == ASMR_PROJECT_VERSION
    assert project.config["project_file_version"] == ASMR_PROJECT_VERSION


def test_legacy_fork_project_migrated(tmpdir):
    project = asr.Project.create(Path(tmpdir, "src.asreview"))

    config_fp = Path(project.project_path, "project.json")
    with open(config_fp) as f:
        config = json.load(f)
    config["project_file_version"] = 4
    with open(config_fp, "w") as f:
        json.dump(config, f)

    migrate_project(project.project_path, 4)

    with open(config_fp) as f:
        config = json.load(f)
    assert config["project_file_version"] == ASMR_PROJECT_VERSION


def test_unmarked_v4_project_refused_by_step(tmpdir):
    project = asr.Project.create(Path(tmpdir, "src.asreview"))

    with sqlite3.connect(str(project.db_path)) as conn:
        conn.execute("PRAGMA application_id = 0")
        conn.commit()

    with pytest.raises(ValueError, match="asmr"):
        _migrate_v4_asmr(project.project_path)


def test_import_unsupported_version_refused(tmpdir):
    project = asr.Project.create(Path(tmpdir, "src.asreview"))

    config_fp = Path(project.project_path, "project.json")
    with open(config_fp) as f:
        config = json.load(f)
    config["project_file_version"] = 5
    with open(config_fp, "w") as f:
        json.dump(config, f)

    export_fp = Path(tmpdir, "src-export.asreview")
    project.export(export_fp)

    projects_dir = Path(tmpdir, "projects")
    projects_dir.mkdir()

    with pytest.raises(ValueError, match="Unsupported project file version"):
        asr.Project.load(export_fp, projects_dir)

    assert list(projects_dir.iterdir()) == []


def test_import_legacy_fork_export(tmpdir):
    project = asr.Project.create(Path(tmpdir, "src.asreview"))

    config_fp = Path(project.project_path, "project.json")
    with open(config_fp) as f:
        config = json.load(f)
    config["project_file_version"] = 4
    with open(config_fp, "w") as f:
        json.dump(config, f)

    export_fp = Path(tmpdir, "src-export.asreview")
    project.export(export_fp)

    projects_dir = Path(tmpdir, "projects")
    projects_dir.mkdir()
    imported = asr.Project.load(export_fp, projects_dir)

    assert detect_version(imported.config) == ASMR_PROJECT_VERSION
