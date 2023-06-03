from core.labeling.cvat.client import CVATClient
from core.utils.aws_utils import get_secret_from_aws_secret_manager

video_uuids = [
    "americold/modesto/F_dock/ch11/americold_modesto_2020-12-23_F_dock_ch11_20201022165433_20201022225738_lower_10530_upper_14100",
]


def get_cvat_for_task_name():
    """Get cvat for task name"""
    credentials = get_secret_from_aws_secret_manager("CVAT_CREDENTIALS")
    cvat_host = "cvat.voxelplatform.com"
    cvat_client = CVATClient(cvat_host, credentials)
    for video_uuid in video_uuids:
        print(cvat_client.tasks_list_by_name(video_uuid)[0].get("id"))


def main():
    """Main"""
    get_cvat_for_task_name()


if __name__ == "__main__":
    main()
