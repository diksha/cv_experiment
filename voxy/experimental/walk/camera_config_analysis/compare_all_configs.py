import json
from argparse import ArgumentParser

from core.execution.utils.graph_config_builder import _flatten_config

parser = ArgumentParser()
parser.add_argument("--original", type=str, required=True)
parser.add_argument("--merged", type=str, required=True)
args = parser.parse_args()

with open(args.original, "r", encoding="utf-8") as file:
    original_configs: dict = json.load(file)

with open(args.merged, "r", encoding="utf-8") as file:
    merged_configs: dict = json.load(file)

config_names = set(original_configs.keys()).union(merged_configs.keys())

if set(original_configs.keys()) == set(merged_configs.keys()):
    print(f"both have the same set of {len(config_names)} camera configs")

for config in config_names:
    flat_original = _flatten_config(original_configs[config])
    flat_merged = _flatten_config(merged_configs[config])

    keys = set(flat_original.keys()).union(flat_merged.keys())
    for key in keys:
        if key not in flat_original:
            print(f"{config} has {key} in merged but not original")
            continue

        if key not in flat_merged:
            print(f"{config} has {key} in original but not merged")
            continue

        if flat_original[key] != flat_merged[key]:
            print(
                f"{config}[{key}] discrepancy:\n\t '{flat_original[key]}' "
                f"in OG vs '{flat_merged[key]}' in merged"
            )

# trunk-ignore(pylint/W0105)
"""
bazel run //experimental/walk/camera_config_analysis:compare_all_configs -- \
    --original=$HOME/voxel/experimental/walk/camera_config_analysis/original_configs.json \
    --merged=$HOME/voxel/experimental/walk/camera_config_analysis/configs_after_merge.json
"""
