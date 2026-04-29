# Copyright 2019-2025 The ASReview Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import time
from pathlib import Path

import asreview as asr
from asreview.models.queriers import TopDown
from asreview.models.stoppers import IsFittable
from asreview.simulation.simulate import Simulate
from asreview.webapp.utils import get_project_path

logger = logging.getLogger("asreview.task_manager")

TRAINING_PROGRESS_FILE = "training_progress.json"


def _read_cycle_data(project):
    return asr.ActiveLearningCycleData(**project.get_model_config())


def _progress_path(project):
    return Path(project.project_path) / TRAINING_PROGRESS_FILE


def _write_progress(project, phase, label, n_records=None, n_labeled=None):
    """Write training progress to a JSON file and log it."""
    data = {
        "phase": phase,
        "label": label,
        "timestamp": time.time(),
    }
    if n_records is not None:
        data["n_records"] = n_records
    if n_labeled is not None:
        data["n_labeled"] = n_labeled

    try:
        _progress_path(project).write_text(json.dumps(data))
    except Exception:
        pass

    logger.info(
        "Training progress [%s]: %s (records=%s, labeled=%s)",
        phase,
        label,
        n_records,
        n_labeled,
    )


def _clear_progress(project):
    """Remove the progress file when training is done."""
    try:
        _progress_path(project).unlink(missing_ok=True)
    except Exception:
        pass


def run_task(project_id, simulation=False):
    project_path = get_project_path(project_id)

    with asr.Project(project_path, project_id=project_id) as project:
        if simulation:
            run_simulation(project)
        else:
            run_model(project)

    return True


def run_model(project):
    n_records = None

    with project.db as db:
        if not db.exist_new_labeled_records:
            return

        labeled = db.get_results_table(columns=["record_id", "label"])
        # Only train a new model if both 0 and 1 labels are available:
        if labeled["label"].value_counts().shape[0] < 2:
            return

        n_records = len(db.input)

    n_labeled = len(labeled)

    _write_progress(
        project, "loading", "Loading model configuration...",
        n_records=n_records, n_labeled=n_labeled,
    )

    try:
        cycle_data = _read_cycle_data(project)

        if cycle_data.feature_extractor:
            try:
                fm = project.get_feature_matrix(cycle_data.feature_extractor)
                cycle = asr.ActiveLearningCycle.from_meta(
                    cycle_data, skip_feature_extraction=True
                )
                _write_progress(
                    project, "feature_extraction",
                    "Feature matrix loaded from cache",
                    n_records=n_records, n_labeled=n_labeled,
                )
            except ValueError:
                _write_progress(
                    project, "feature_extraction",
                    "Extracting features...",
                    n_records=n_records, n_labeled=n_labeled,
                )
                cycle = asr.ActiveLearningCycle.from_meta(cycle_data)
                fm = cycle.transform(project.db.input.get_df())
                project.add_feature_matrix(fm, cycle.feature_extractor.name)
                _write_progress(
                    project, "feature_extraction",
                    "Feature extraction complete",
                    n_records=n_records, n_labeled=n_labeled,
                )
        else:
            cycle = asr.ActiveLearningCycle.from_meta(cycle_data)
            fm = project.db.input.get_df().values

        if cycle.classifier is not None:
            _write_progress(
                project, "fitting",
                "Training classifier...",
                n_records=n_records, n_labeled=n_labeled,
            )
            cycle.fit(
                fm[labeled["record_id"].values],
                labeled["label"].values,
            )
            _write_progress(
                project, "fitting",
                "Classifier training complete",
                n_records=n_records, n_labeled=n_labeled,
            )

        _write_progress(
            project, "ranking",
            "Ranking records...",
            n_records=n_records, n_labeled=n_labeled,
        )
        ranked_record_ids = cycle.rank(fm)

        _write_progress(
            project, "saving",
            "Saving results...",
            n_records=n_records, n_labeled=n_labeled,
        )
        with project.db as db:
            db.add_last_ranking(
                ranked_record_ids,
                cycle_data.classifier if cycle_data.classifier is not None else None,
                cycle_data.querier,
                cycle_data.balancer if cycle_data.balancer is not None else None,
                cycle_data.feature_extractor
                if cycle_data.feature_extractor is not None
                else None,
                len(labeled),
            )

        project.remove_review_error()
        _clear_progress(project)

    except Exception as err:
        _write_progress(
            project, "error", str(err),
            n_records=n_records, n_labeled=n_labeled,
        )
        project.set_review_error(err)
        raise err


def run_simulation(project):
    with project.db as db:
        priors = db.get_priors()["record_id"].tolist()

    cycles = [
        asr.ActiveLearningCycle(
            querier=TopDown(),
            stopper=IsFittable(),
        ),
        asr.ActiveLearningCycle.from_meta(_read_cycle_data(project)),
    ]

    sim = Simulate(
        project.db.input.get_df(),
        project.db.input["included"],
        cycles,
        print_progress=False,
        groups=project.db.input.get_groups(),
    )
    try:
        sim.label(priors)
        sim.review()
    except Exception as err:
        project.set_review_error(err)
        raise err

    project.update_review(status="finished")
    sim.to_sql(project.db_path)
