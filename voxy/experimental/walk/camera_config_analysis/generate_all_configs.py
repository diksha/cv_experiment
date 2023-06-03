import json
import os
from argparse import ArgumentParser

from core.execution.utils.graph_config_builder import GraphConfigBuilder
from core.utils.yaml_jinja import load_yaml_with_jinja

parser = ArgumentParser()
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--configs_dir", type=str, required=True)
parser.add_argument("--default_config_path", type=str)
parser.add_argument("--env_config_path", type=str)
args = parser.parse_args()

# Define default and env to perform merge, otherwise just use config input
if (args.default_config_path is None) != (args.env_config_path is None):
    print("must define all or none of {default_config_path, env_config_path}")

do_merge = args.default_config_path is not None

all_config_files = [
    os.path.join(dir_path, fname)
    for dir_path, _, file_names in os.walk(args.configs_dir)
    for fname in file_names
    if fname.endswith("cha.yaml")
]

if do_merge:
    env_config = load_yaml_with_jinja(args.env_config_path)
    default_config = load_yaml_with_jinja(args.default_config_path)


all_configs = {}
for path in all_config_files:
    config_name = (
        path.removeprefix(args.configs_dir)
        .removesuffix("cha.yaml")
        .strip("/")
        .replace("/", "_")
    )

    camera_config = load_yaml_with_jinja(path)
    if do_merge:
        builder = GraphConfigBuilder()
        builder.apply(default_config, "default")
        builder.apply(camera_config, "camera")
        builder.apply(env_config, "env")
        all_configs[config_name] = builder.get_config()
    else:
        all_configs[config_name] = camera_config

with open(args.output_path, "w+", encoding="utf-8") as file:
    json.dump(all_configs, file)


# trunk-ignore(pylint/W0105)
"""
bazel run //experimental/walk/camera_config_analysis:generate_all_configs -- --configs_dir=$HOME/voxel/configs/cameras --output_path=$HOME/voxel/experimental/walk/camera_config_analysis/original_configs.json
bazel run //experimental/walk/camera_config_analysis:generate_all_configs -- --configs_dir=$HOME/voxel/configs/cameras --output_path=$HOME/voxel/experimental/walk/camera_config_analysis/configs_after_merge.json --default_config_path=$HOME/voxel/configs/graphs/default.yaml --env_config_path=$HOME/voxel/configs/graphs/production/environment/production.yaml
"""
