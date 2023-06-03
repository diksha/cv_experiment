from sematic.resolvers.cloud_resolver import CloudResolver

from core.ml.data.curation.lib.lightly_worker import run_lightly_worker
from core.utils.yaml_jinja import load_yaml_with_jinja


def main():
    """Testing calculating frame sequences"""
    run_lightly_worker(
        dataset_id="doors_cropped_americold_ontario_0002_cha_2",
        dataset_name="doors_cropped_americold_ontario_0002_cha_2",
        input_directory="doors/cropped/americold/ontario/0002/cha",
        output_directory="doors/cropped/americold/ontario/0002/cha",
        config=load_yaml_with_jinja(
            "core/ml/data/curation/configs/DOOR_STATE.yaml"
        ),
    ).resolve(CloudResolver())


if __name__ == "__main__":
    main()
