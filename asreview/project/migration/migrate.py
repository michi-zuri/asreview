import json
import logging
import shutil
import sqlite3
import tempfile
from pathlib import Path

from asreview.database.fork_migrations import ASMR_PROJECT_VERSION
from asreview.project.migration.v1v2 import _migrate as _migrate_v1v2
from asreview.project.migration.v2v3 import _migrate as _migrate_v2v3
from asreview.project.migration.v2v3 import _validate as _validate_v2v3
from asreview.project.migration.v3v4 import _migrate as _migrate_v3v4
from asreview.project.migration.v3v4 import _validate as _validate_v3v4


__all__ = [
    "detect_version",
    "migrate_project",
    "needs_migration",
    "is_supported_version",
    "UPSTREAM_VERSIONS",
]

# Upstream ASReview project file versions, each migrating to the next.
UPSTREAM_VERSIONS = (1, 2, 3)

# The migration chain, ending at the asmr format. A plain 4 is a legacy
# fork project.json (what 10d2 and earlier wrote); it migrates to asmr too.
NEXT_VERSION = {
    1: 2,
    2: 3,
    3: ASMR_PROJECT_VERSION,
    4: ASMR_PROJECT_VERSION,
}


def needs_migration(version):
    """Return True if ``version`` is a project format that can migrate to asmr."""
    return version in NEXT_VERSION


def is_supported_version(version):
    """Return True if ``version`` is an asmr project or one that can migrate."""
    return version == ASMR_PROJECT_VERSION or needs_migration(version)


def is_fork_database(db_fp):
    """Return True if the database at ``db_fp`` carries the asmr mark."""
    from asreview.database.fork_migrations import FORK_APPLICATION_ID

    conn = sqlite3.connect(f"file:{db_fp}?mode=ro", uri=True)
    try:
        app_id = conn.execute("PRAGMA application_id").fetchone()[0]
    finally:
        conn.close()
    return app_id == FORK_APPLICATION_ID


def detect_version(project_config):
    """Detect the version of a project from it's config.

    Arguments
    ---------
    project_config : dict
        Project config.

    Returns
    -------
    int
        Project file version; -1 for an asmr project (its schema state is
        in the results.db ledger).
    """
    # If the project file version is explicitly specified use that value.
    if "project_file_version" in project_config:
        return project_config["project_file_version"]
    # Otherwise use the ASReview major version.
    return int(project_config["version"][0])


def migrate_project(folder, src_version, dst_version=ASMR_PROJECT_VERSION):
    """Migrate a project.

    Parameters
    ----------
    folder: str
        The folder of the project to migrate
    src_version: int
        The project file version to migrate from.
    dst_version: int, optional
        The version to migrate to; only the asmr format (-1) is supported.

    Returns
    -------
    None
    """
    if not needs_migration(src_version):
        raise ValueError(f"Project version {src_version} cannot be migrated.")
    if dst_version != ASMR_PROJECT_VERSION:
        raise ValueError("Destination version should be the asmr format (-1).")

    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Project folder {folder} does not exist.")

    if not Path(folder, "project.json").exists():
        raise FileNotFoundError(
            f"Project file {Path(folder, 'project.json')} does not exist."
        )

    # try to get the project id from the project.json file
    with open(Path(folder, "project.json")) as f:
        project_config = json.load(f)
        project_id = project_config.get("id", "unknown")

    if src_version == 1 and _is_empty_v1(folder):
        shutil.rmtree(folder)
        return

    if _is_empty_v2(folder, src_version):
        logging.warning(
            f"Skipping migration of project {project_id}: "
            "no review configured. This project cannot be migrated."
        )
        return

    current_version = src_version
    while current_version != dst_version:
        try:
            _migrate_project_one_version(folder, current_version)
            current_version = NEXT_VERSION[current_version]
        except Exception as e:
            logging.exception(
                f"Failed to migrate project {folder} from {current_version} to "
                f"{NEXT_VERSION[current_version]}:\n{e}"
            )
            raise Exception(f"Failed to upgrade project {project_id}.") from e


def _migrate_project_one_version(folder, current_version):
    if current_version == 1:
        migrate = _migrate_v1v2
        validate = None
    elif current_version == 2:
        migrate = _migrate_v2v3
        validate = _validate_v2v3
    elif current_version == 3:
        migrate = _migrate_v3v4
        validate = _validate_v3v4
    elif current_version == 4:
        migrate = _migrate_v4_asmr
        validate = _validate_v4_asmr
    else:
        raise ValueError("Invalid current version.")

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copytree(
            folder,
            Path(tmpdir) / "tmp_project",
            ignore=shutil.ignore_patterns("*.lock"),
        )

        # Only project.json changes in the 4 → asmr step, so no legacy copy
        # is needed; the .backup swap below covers failure.
        if current_version != 4:
            shutil.copytree(
                folder,
                Path(tmpdir) / "tmp_project" / f"legacy_v{current_version}",
                ignore=shutil.ignore_patterns("*.lock"),
            )

        next_version = NEXT_VERSION[current_version]
        next_label = (
            "asmr" if next_version == ASMR_PROJECT_VERSION else f"v{next_version}"
        )
        logging.info(
            f"Upgrading project {folder} from v{current_version} to {next_label}."
        )
        migrate(Path(tmpdir) / "tmp_project")
        if validate is not None:
            validate(Path(tmpdir) / "tmp_project")

        backup = folder.with_name(folder.name + ".backup")
        shutil.move(folder, backup)
        try:
            shutil.move(Path(tmpdir) / "tmp_project", folder)
        except Exception:
            shutil.move(backup, folder)
            raise
        shutil.rmtree(backup)


def _migrate_v4_asmr(project):
    """Migrate a legacy fork project (project_file_version 4) to asmr.

    Only the project.json field changes; the results.db already carries the
    asmr mark. A project without a marked results.db is refused (an unmarked
    4 is a future upstream project, not a fork project).
    """
    project = Path(project)
    results_db = Path(project, "results.db")
    if not results_db.exists() or not is_fork_database(results_db):
        raise ValueError(
            "Project version 4 without the asmr mark: not an asmr project."
        )

    config_fp = Path(project, "project.json")
    with open(config_fp) as f:
        project_config = json.load(f)
    project_config["project_file_version"] = ASMR_PROJECT_VERSION
    with open(config_fp, "w") as f:
        json.dump(project_config, f)


def _validate_v4_asmr(project):
    """Validate a project migrated from a legacy 4 to the asmr format."""
    project = Path(project)
    config_fp = Path(project, "project.json")
    if not config_fp.exists():
        raise ValueError("Migrated project is missing project.json.")

    with open(config_fp) as f:
        config = json.load(f)

    if config.get("project_file_version") != ASMR_PROJECT_VERSION:
        raise ValueError(
            f"Expected project file version {ASMR_PROJECT_VERSION}, "
            f"got {config.get('project_file_version')}."
        )


def _is_empty_v2(folder, current_version):
    """Check if a project at v2 has no reviews configured.

    A v2 project without reviews cannot be migrated to v3 because the review
    data (results.db, data_store.db) does not exist. This should not happen in
    practice, but if it does we skip the project gracefully.

    Parameters
    ----------
    folder : str | Path
        The folder of the project to check.
    current_version : int
        The current version of the project (or the source version if
        the project has already been migrated from v1 to v2).

    Returns
    -------
    bool
        True if the project is at v2 with no reviews, False otherwise.
    """
    if current_version != 2:
        return False

    with open(Path(folder, "project.json")) as f:
        config = json.load(f)

    reviews = config.get("reviews", [])
    return len(reviews) == 0


def _is_empty_v1(folder):
    """Check if a v1 project is empty (no reviews configured).

    Parameters
    ----------
    folder : str | Path
        The folder of the v1 project to check.

    Returns
    -------
    bool
        True if the project has no reviews, False otherwise.
    """
    with open(Path(folder, "project.json")) as f:
        config = json.load(f)

    return len(config.get("reviews", [])) == 0
