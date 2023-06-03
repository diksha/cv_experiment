import argparse
import base64
import os
import sys
import time
from datetime import datetime

import boto3
from loguru import logger

ec2_client = boto3.client("ec2", region_name="us-west-2")
route53_client = boto3.client("route53", region_name="us-west-2")


class DevBox:
    def __init__(
        self,
        target_user,
        instance_id,
        instance_type,
        instance_state,
        public_ip,
        private_ip,
        volume_id,
    ):
        self.target_user = target_user
        self.instance_id = instance_id
        self.instance_type = instance_type
        self.instance_state = instance_state
        self.public_ip = public_ip
        self.private_ip = private_ip
        self.volume_id = volume_id

    def __repr__(self):
        """
        Returns:
            str: string output devbox url
        """
        return f"{self.target_user}.devbox.voxelplatform.com ->  \
            {str.upper(self.instance_state)} {self.instance_id}[{self.instance_type}] \
            IP: {self.private_ip}"

    def boto_instance(self):
        """Boto EC2 Instance

        Returns:
            boto3_ec2_instance: returns boto3 ec2 instance
        """
        return boto3.resource("ec2", region_name="us-west-2").Instance(
            self.instance_id
        )

    def boto_volume(self):
        """Boto EC2 Volume

        Returns:
            boto3_ec2_volume: returns boto3 ec2 volume
        """
        return boto3.resource("ec2", region_name="us-west-2").Volume(
            self.volume_id
        )


def __user_data__(target_user: str, ssh_key):
    """User Data.

    Args:
        target_user (str): username
        ssh_key (str): ssh public key

    Returns:
        str: userdata for the instance
    """
    if not ssh_key or ssh_key == "default":
        raise_error("ssh key missing")
    data = ssh_key.encode("ascii")
    base_64_value = base64.b64encode(data).decode("utf-8")
    user_data = f"""#!/bin/bash
set -euo pipefail
export DEVBOX_USER_ID=20001
export DEVBOX_GROUP=voxel
export DEVBOX_GROUP_ID=10001
export DEVBOX_USER={target_user}
export DEVBOX_USER_PUBLIC_KEY_BASE_64={base_64_value}
/devbox.sh
hostnamectl set-hostname {target_user}-devbox.local
"""
    return user_data


def find_devbox(target_user: str):
    """Find existing devbox for user.

    Args:
        target_user (str): user

    Returns:
        Devbox: devbox object
    """
    reservations = ec2_client.describe_instances(
        Filters=[
            {
                "Name": "tag:Name",
                "Values": [f"devbox-{target_user}"],
            },
        ]
    ).get("Reservations")

    for reservation in reservations:
        if len(reservation["Instances"]) > 1:
            raise_error(
                "Multiple instances found for given user. Please get it fixed from Platform team"
            )
        instance = reservation["Instances"][0]
        volume_id = (
            instance["BlockDeviceMappings"][0]["Ebs"]["VolumeId"]
            if len(instance["BlockDeviceMappings"]) > 0
            else None
        )
        return DevBox(
            target_user,
            instance["InstanceId"],
            instance["InstanceType"],
            instance["State"]["Name"],
            instance.get("PublicIpAddress", None),
            instance.get("PrivateIpAddress", None),
            volume_id,
        )


def __ensure_route53_records__(devbox: DevBox):
    """Add record into Route53.

    Args:
        devbox (DevBox): devbox instance object
    Returns:
        Response: response object
    """
    response = route53_client.change_resource_record_sets(
        HostedZoneId="Z07178513JMVAX79KI1YZ",
        ChangeBatch={
            "Comment": f"DevBox for {devbox.target_user}",
            "Changes": [
                {
                    "Action": "UPSERT",
                    "ResourceRecordSet": {
                        "Name": f"{devbox.target_user}.devbox.voxelplatform.com",
                        "Type": "A",
                        "TTL": 300,
                        "ResourceRecords": [
                            {"Value": devbox.private_ip},
                        ],
                    },
                },
            ],
        },
    )
    return response


def __create_devbox__(target_user: str, instance_type: str, ssh_key: str):
    """Create devbox

    Args:
        target_user (str): user
        instance_type (str): instance type
        ssh_key (str): ssh public key

    Returns:
        Devbox: created devbox
    """
    ec2_client.run_instances(
        MaxCount=1,
        MinCount=1,
        LaunchTemplate={"LaunchTemplateId": "lt-03e9065e5655509a0"},
        InstanceType=instance_type,
        UserData=__user_data__(target_user, ssh_key),
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": f"devbox-{target_user}"}],
            }
        ],
    )
    time.sleep(10)
    devbox = find_devbox(target_user)
    devbox.boto_instance().wait_until_running()
    return find_devbox(target_user)


def __start_existing_devbox__(
    existing_devbox: DevBox, expected_instance_type: str
):
    """Start existing devbox

    Args:
        existing_devbox (DevBox): devbox object
        expected_instance_type (str): instance type

    Returns:
        Devbox: started devbox object
    """
    ec2_client.modify_instance_attribute(
        InstanceId=existing_devbox.instance_id,
        InstanceType={"Value": expected_instance_type},
    )
    ec2_client.start_instances(InstanceIds=[existing_devbox.instance_id])
    existing_devbox.boto_instance().wait_until_running()
    return find_devbox(existing_devbox.target_user)


def stop_devbox(target_user: str):
    """Stop devbox

    Args:
        target_user (str): user
    """
    existing_devbox = find_devbox(target_user)
    if existing_devbox is None:
        logger.info("No DevBox found. Nothing to do.")
    elif existing_devbox.instance_state == "running":
        logger.info(
            f"A running DevBox was found for {target_user} with id {existing_devbox.instance_id}"
        )
        ec2_client.stop_instances(
            InstanceIds=[existing_devbox.instance_id], Force=True
        )
        logger.info("Waiting for instance to stop...")
        existing_devbox.boto_instance().wait_until_stopped()
        logger.info("DevBox was stopped successfully!")


def delete_devbox(target_user: str):
    """Delete devbox

    Args:
        target_user (str): user
    """
    existing_devbox = find_devbox(target_user)
    if existing_devbox is None:
        raise_error("No DevBox found. Nothing to do.")
    ec2_client.terminate_instances(InstanceIds=[existing_devbox.instance_id])
    logger.info("Waiting for instance to terminate...")
    existing_devbox.boto_instance().wait_until_terminated()
    logger.info("DevBox was terminated successfully!")
    existing_devbox.boto_instance().create_tags(
        Tags=[
            {"Key": "Name", "Value": f"devbox-terminated-{target_user}"},
            {
                "Key": "TermDate",
                "Value": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            },
        ]
    )


def start_devbox(
    target_user: str, instance_size: str, provide_gpu: bool, ssh_key: str
):
    """Starts a devbox for the user

    Args:
        target_user (str): user
        instance_size (str): instance size
        provide_gpu (bool): whether to add gpu or not
        ssh_key (str): ssh public key
    """
    devbox = find_devbox(target_user)
    expected_instance_type = (
        f'{"g4dn" if provide_gpu else "t3"}.{instance_size}'
    )
    if devbox is None:
        logger.info("Devbox was not found. Starting it ...")
        devbox = __create_devbox__(
            target_user, expected_instance_type, ssh_key
        )
    else:
        if devbox.instance_state == "running":
            logger.info("Devbox is already running. Nothing to do!")
        elif devbox.instance_state == "stopped":
            devbox = __start_existing_devbox__(devbox, expected_instance_type)
        elif devbox.instance_state == "terminated":
            logger.info(
                "A terminated Devbox was found. Renaming it and creating a fresh one ..."
            )
            devbox = __create_devbox__(
                target_user, expected_instance_type, ssh_key
            )
        else:
            raise_error(
                f"Devbox was found in an unexpected state {devbox.instance_state}. \
                    Please try again in 20 minutes or contact Platform team thereafter."
            )
    if devbox is None:
        raise_error(
            "No DevBox seems to be created. Please contact Platform team"
        )
    __post_start_devbox__(devbox)
    logger.info("Devbox is running successfully. Details are: ")
    logger.info(devbox)
    logger.info(
        "You may SSH into the DevBox or connect to it using VSCode. \
            It may take couple minutes to be online. \
            Please follow the guide in ClickUp."
    )


def __post_start_devbox__(devbox: DevBox):
    """Steps after devbox started

    Args:
        devbox (DevBox): devbox object
    """
    __ensure_route53_records__(devbox)
    devbox.boto_instance().create_tags(
        Tags=[{"Key": "map-migrated", "Value": "d-server-00swbp99drezfh"}]
    )


def __is_outdated__(devbox: DevBox):
    """Check if outdata

    Args:
        devbox (DevBox): devbox object

    Returns:
        bool: is_outdated
    """
    if devbox.public_ip is not None and devbox.public_ip != "":
        return True
    if devbox.boto_volume().size != 500:
        return True
    return False


def recreate_devbox(target_user: str, ssh_key: str, force_recreate=False):
    """Recreates the devbox

    Args:
        target_user (str): user
        ssh_key (str): ssh public key
        force_recreate (bool, optional): whether to force it, defaults to false.
    """
    old_devbox = find_devbox(target_user)
    if old_devbox is None:
        raise_error("No DevBox found. Cannot recreate!")
    if __is_outdated__(old_devbox):
        stop_devbox(target_user)
        if old_devbox.volume_id is None:
            raise_error("No volume attached to this DevBox. Error!")
        old_devbox.boto_volume().create_tags(
            Tags=[
                {"Key": "MigrationTargetUser", "Value": target_user},
                {
                    "Key": "MigrationDetachDate",
                    "Value": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                },
            ]
        )
        old_devbox.boto_instance().detach_volume(VolumeId=old_devbox.volume_id)
        ec2_client.modify_volume(VolumeId=old_devbox.volume_id, Size=500)
        logger.info(
            f"Old DevBox stopped and volume {old_devbox.volume_id} detached"
        )
        old_devbox.boto_instance().create_tags(
            Tags=[
                {
                    "Key": "Name",
                    "Value": f"devbox-to-be-terminated-{target_user}",
                }
            ]
        )
        new_devbox = __create_devbox__(
            target_user, old_devbox.instance_type.replace("m5", "t3"), ssh_key
        )
        stop_devbox(target_user)
        new_devbox.boto_instance().detach_volume(VolumeId=new_devbox.volume_id)
        logger.info(
            f"New DevBox stopped and volume {new_devbox.volume_id} detached"
        )
        new_devbox.boto_instance().attach_volume(
            Device="/dev/sda1", VolumeId=old_devbox.volume_id
        )
        __start_existing_devbox__(new_devbox, new_devbox.instance_type)
        logger.info(
            f"New DevBox started with old volume {old_devbox.volume_id}"
        )
        __ensure_route53_records__(new_devbox)
        ec2_client.delete_volume(VolumeId=new_devbox.volume_id)
        logger.info("Waiting for DNS to propagate!")
        time.sleep(300)
    else:
        logger.info(
            "DevBox is not outdated and force is not specified. Skipping."
        )


def raise_error(message: str):
    """Raise error with the given message.

    Args:
        message (str): error message
    """
    logger.info(f"ERROR: {message}")
    sys.exit(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Devbox Parser")
    parser.add_argument("--action", type=str, required=True)
    parser.add_argument("--provide_gpu", type=str, required=True)
    parser.add_argument("--instance_size", type=str, required=True)
    parser.add_argument(
        "--ssh_public_key", nargs="+", required=False, default=[]
    )
    args = parser.parse_args()
    build_user_id = os.environ.get("BUILDKITE_BUILD_CREATOR_EMAIL")
    target_action = args.action.upper()
    if build_user_id is None:
        raise_error(
            "BUILDKITE_BUILD_CREATOR_EMAIL is fetched from Buildkite.\
             This should be user email ideally!"
        )
    t_user = build_user_id.split("@")[0]
    logger.info(f"DevBox Services for {t_user} !")

    if target_action == "START":
        input_provide_gpu = args.provide_gpu == "true"
        provided_instance_size = args.instance_size
        start_devbox(
            t_user,
            provided_instance_size,
            input_provide_gpu,
            " ".join(args.ssh_public_key),
        )
    elif target_action == "RESTART":
        stop_devbox(t_user)
        input_provide_gpu = args.provide_gpu == "true"
        provided_instance_size = args.instance_size
        start_devbox(
            t_user,
            provided_instance_size,
            input_provide_gpu,
            " ".join(args.ssh_public_key),
        )
    elif target_action == "STOP":
        stop_devbox(t_user)
    elif target_action == "STATUS":
        logger.info(find_devbox(t_user))
    elif target_action == "DESTROY":
        delete_devbox(t_user)
    elif target_action == "RECREATE":
        recreate_devbox(
            t_user,
            os.environ.get("FORCE", "false") == "true",
            " ".join(args.ssh_public_key),
        )
    elif target_action == "FIX":
        found_devbox = find_devbox(t_user)
        __post_start_devbox__(found_devbox)
    else:
        raise_error("Invalid target action!")
