import sqlite3
from importlib.metadata import entry_points


from asreview import extensions


def read_tags_data(db):
    """Read tag group definitions from ``tag_groups`` + ``tag_options`` tables.

    Parameters
    ----------
    db : asreview.database.database.Database
        An open database connection.

    Returns
    -------
    list[dict] | None
        Tag groups in the same nested format the API expects, or ``None`` if
        the ``tag_groups`` table does not exist (e.g., a read-only legacy
        project that has not been migrated yet).
    """
    try:
        cur = db._conn.cursor()
        rows = cur.execute(
            "SELECT g.group_id, g.export_name, g.label_name, "
            "g.required_for_relevant, g.required_for_irrelevant, "
            "g.all_required, g.single, g.input_helper_text, g.sorted_at, "
            "o.option_id, o.export_name, o.label_name, "
            "o.free_text_enabled, o.free_text_required, o.sorted_at "
            "FROM tag_groups g "
            "LEFT JOIN tag_options o ON o.group_id = g.group_id "
            "ORDER BY g.sorted_at, g.group_id, o.sorted_at, o.option_id"
        ).fetchall()
    except sqlite3.OperationalError:
        return None

    groups = {}
    for row in rows:
        gid = row[0]
        if gid not in groups:
            groups[gid] = {
                "id": gid,
                "export": row[1],
                "label": row[2],
                "required_relevant": bool(row[3]),
                "required_irrelevant": bool(row[4]),
                "require_all": bool(row[5]),
                "single_select": bool(row[6]),
                "input_helper_text": row[7] or "",
                "sorted_at": row[8],
                "values": [],
            }
        if row[9] is not None:
            groups[gid]["values"].append(
                {
                    "id": row[9],
                    "export": row[10],
                    "label": row[11],
                    "free_text": bool(row[12]),
                    "free_text_required": bool(row[13]),
                    "sorted_at": row[14],
                }
            )
    return list(groups.values())


def read_lists_data(project):
    """Read list configuration from the ``list_containers`` database table.

    Parameters
    ----------
    project : asreview.project.Project
        An open project.

    Returns
    -------
    list[dict] | None
        List definitions in the same nested format the API expects, or
        ``None`` if the ``list_containers`` table does not exist yet.
    """
    try:
        with project.db as db:
            cur = db._conn.cursor()
            rows = cur.execute(
                "SELECT list_id, name, required_for_relevant, description, sorted_at "
                "FROM list_containers ORDER BY sorted_at, list_id"
            ).fetchall()
    except sqlite3.OperationalError:
        return None

    return [
        {
            "id": row[0],
            "name": row[1],
            "required_for_relevant": bool(row[2]),
            "input_helper_text": row[3] or "",
            "sorted_at": row[4],
        }
        for row in rows
    ]


def add_id_to_tags(group):
    if "values" not in group:
        return group

    for i, _ in enumerate(group["values"]):
        if "id" in group["values"][i]:
            continue

        group["values"][i]["id"] = i

    return group


def get_dist_extensions_metadata():
    """Get all distributions with models."""
    entries = entry_points(group="asreview.models", name="_metadata")

    all_metadata = {}

    for e in entries:
        try:
            metadata = e.load()

            if not isinstance(metadata, dict):
                raise TypeError(
                    f"Metadata for {e.name} is not a dictionary: {type(metadata)}"
                )

            for key, value in metadata.items():
                if key in all_metadata and isinstance(all_metadata[key], dict):
                    all_metadata[key].update(value)
                else:
                    all_metadata[key] = value

        except Exception:
            continue

    return all_metadata


def get_all_model_components():
    model_components = {
        "balancers": [],
        "classifiers": [],
        "feature_extractors": [],
        "queriers": [],
    }

    entry_points_per_submodel = [
        extensions("models.balancers"),
        extensions("models.classifiers"),
        extensions("models.feature_extractors"),
        extensions("models.queriers"),
    ]

    metadata = get_dist_extensions_metadata()

    for entries, key in zip(entry_points_per_submodel, model_components.keys()):
        for e in entries:
            try:
                label = metadata[key][e.name]["label"]
            except KeyError:
                label = e.name
            except Exception as err:
                raise Exception(f"Failed to read metadata: {err}")

            model_components[key].append(
                {
                    "name": e.name,
                    "label": label,
                }
            )

    return model_components
