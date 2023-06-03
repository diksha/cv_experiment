import os

import yaml

from core.utils.yaml_jinja import resolve_jinja_config

CAMERA_CONFIG_MAIN_DIR = "/home/walker/voxel/configs/cameras/"
CAMERA_MANIFEST_FILE = "/home/walker/voxel/configs/cameras/cameras"

# trunk-ignore-all(pylint/W9011)
# trunk-ignore-all(pylint/W9012)
# trunk-ignore-all(pylint/C0116)


def load_config(path):
    return yaml.safe_load(resolve_jinja_config(path))


def get_all_config_paths(manifest_file):
    with open(manifest_file, "r", encoding="utf-8") as file:
        return file.read().splitlines()


def get_all_config_paths_dir_search(root_dir):
    paths = [
        os.path.join(dir_path, fname)
        for dir_path, _, file_names in os.walk(root_dir)
        if "experimental" not in dir_path and "template" not in dir_path
        for fname in file_names
        if fname.endswith(".yaml") and not fname.endswith("_calibration.yaml")
    ]
    print(f"loaded {len(paths)} from {root_dir}")
    # for p in paths: print(p)
    return paths


def set_config_values(config_tracker, config, fname):
    for key in config:
        value = config[key]
        if isinstance(value, list):
            value = ",".join(value)

        config_tracker.setdefault(key, {})
        if isinstance(value, dict):
            set_config_values(config_tracker[key], value, fname)
        else:
            if isinstance(value, list):
                print(key, value)
            config_tracker[key].setdefault(value, [])
            config_tracker[key][value].append(fname)


def get_value_tracker():
    value_tracker = {}
    config_paths = get_all_config_paths(CAMERA_MANIFEST_FILE)

    for config_path in config_paths:
        config = load_config(config_path)
        set_config_values(value_tracker, config, config_path)

    return value_tracker


singly_valued = []
attr_stats = {}


def tracker_insight(tracker_dict, base_attrs=None):
    if base_attrs is None:
        base_attrs = []

    for key, value in tracker_dict.items():
        attr = ".".join(base_attrs + [key])
        value: dict = value
        is_attribute = any(map(lambda x: isinstance(x, list), value.values()))
        if is_attribute:
            n_distinct_values = len(value.values())
            n_appearances = 0
            max_occur = 0
            for _, files in value.items():
                n_appearances += len(files)
                if len(files) > max_occur:
                    max_occur = len(files)

            attr_stats[attr] = (
                n_appearances,
                n_distinct_values,
                max_occur / n_appearances,
            )
        else:
            tracker_insight(value, base_attrs + [key])


tracker = get_value_tracker()
tracker_insight(tracker)


print(tracker["state"]["add_epoch_time"])
insight_tuples = map(
    lambda x: (x[0], x[1][0], x[1][1], x[1][2]), attr_stats.items()
)
insight_tuples = sorted(insight_tuples, key=lambda x: (x[3], x[1], x[2]))


for stats in insight_tuples:
    attribute, appeared, distinct, bias = stats

    print(
        f"{attribute.ljust(70)} appeared {appeared} times with \
{distinct} distinct values and bias of {bias*100:.2f}%"
    )
