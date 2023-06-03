#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import ast
import json

from loguru import logger

from core.utils.perception_portal_graphql_session import (
    PerceptionPortalSession,
)


def get_doors_from_camera_config(
    camera_uuid: str, config_version: int = 1
) -> list:
    """Get doors from the specified camera config.

    Args:
        camera_uuid (str): Eg. americold/ontario/0005/cha
        config_version (int, optional): config version for the specified camera uuid. Defaults to 1.

    Returns:
        list: The door coordinates.
    """
    with PerceptionPortalSession("PROD") as perception_portal_session:
        door_query = f'{{cameraConfigNew(uuid: "{camera_uuid}", \
            version: {config_version}) {{doors}} }}'
        response = perception_portal_session.session.post(
            f"{perception_portal_session.host}/graphql/",
            json={"query": door_query},
            headers=perception_portal_session.headers,
        )
        if response.status_code == 200:
            query_result = json.loads(response.text)["data"]["cameraConfigNew"]
            return ast.literal_eval(query_result["doors"])
        return []


def does_organization_exist(
    portal_env: str,
    organization_key: str,
) -> bool:
    """Checks whether the specified organization exists or not.

    Args:
        portal_env (str): PROD or INTERNAL
        organization_key (str): organization uuid

    Returns:
        bool: True if organization exists
    """
    with PerceptionPortalSession(portal_env) as perception_portal_session:
        query = f'{{organization(organizationKey: "{organization_key}") {{key, id, pk}} }}'
        query_results = perception_portal_session.session.post(
            f"{perception_portal_session.host}/graphql/",
            json={"query": query},
            headers=perception_portal_session.headers,
        )
        org_info = json.loads(query_results.text)["data"]["organization"]
        return org_info is not None


def generate_organization(
    portal_env: str,
    organization_key: str,
    organization_name: str,
    time_zone: str = "US/Pacific",
) -> dict:
    """If the organization does not exist, we use this function to make a new organization.

    Args:
        portal_env (str): PROD or INTERNAL
        organization_key (str): Eg. uscold
        organization_name (str): organization uuid
        time_zone (str, optional): Defaults to "US/Pacific".

    Returns:
        dict: updates the portal with mutation
    """
    with PerceptionPortalSession(portal_env) as perception_portal_session:
        command = f'organizationCreate( \
            organizationKey: "{organization_key}", \
            organizationName: "{organization_name}", \
            timeZone: "{time_zone}")'
        response = "organization {key, name}"
        mutation = f"mutation {{ {command} {{ {response} }} }}"
        mutation_results = perception_portal_session.session.post(
            f"{perception_portal_session.host}/graphql/",
            json={"query": mutation},
            headers=perception_portal_session.headers,
        )
        logger.info(f"MUTATION RESULTS: {mutation_results.text}")
        return json.loads(mutation_results.text)["data"]["organizationCreate"]


def does_zone_exist_for_organization(
    portal_env: str,
    organization_key: str,
    zone_key: str,
) -> bool:
    """Checks if the specified zone exists for an organization

    Args:
        portal_env (str): PROD or INTERNAL
        organization_key (str): organization uuid
        zone_key (str): zone uuid

    Returns:
        bool: True if zone exists for an organization
    """
    with PerceptionPortalSession(portal_env) as perception_portal_session:
        query = f'{{organization(organizationKey: "{organization_key}") {{ sites {{ key }} }} }}'
        query_results = perception_portal_session.session.post(
            f"{perception_portal_session.host}/graphql/",
            json={"query": query},
            headers=perception_portal_session.headers,
        )
        org_info = json.loads(query_results.text)["data"]["organization"]
        if not org_info:
            logger.error(f"ORGANIZATION DOES NOT EXIST: {organization_key}")
            return False

        zones = org_info["sites"]
        for zone in zones:
            if zone["key"] == zone_key:
                logger.info(
                    f"FOUND ZONE, ORGANIZATION: {organization_key}, ZONE: {zone_key}"
                )
                return True

        logger.info(
            f"ZONE DOES NOT EXIST, ORGANIZATION: {organization_key}, ZONE: {zone_key}"
        )
        return False


def generate_zone_for_organization(
    portal_env: str,
    organization_key: str,
    zone_key: str,
    zone_name: str,
    zone_type: str,
) -> dict:
    """If zone does not exist for an organization, we generate a new zone.

    Args:
        portal_env (str): PROD or INTERNAL
        organization_key (str): organization uuid
        zone_key (str): zone uuid
        zone_name (str): Eg. ontario, modesto
        zone_type (str): site

    Returns:
        dict: _description_
    """
    with PerceptionPortalSession(portal_env) as perception_portal_session:
        command = f'zoneCreate( \
            organizationKey: "{organization_key}", \
            zoneKey: "{zone_key}", \
            zoneName: "{zone_name}", \
            zoneType: "{zone_type}")'
        response = "zone {key, name}"
        mutation = f"mutation {{ {command} {{ {response} }} }}"
        mutation_results = perception_portal_session.session.post(
            f"{perception_portal_session.host}/graphql/",
            json={"query": mutation},
            headers=perception_portal_session.headers,
        )
        logger.info(f"MUTATION RESULTS: {mutation_results.text}")
        return json.loads(mutation_results.text)["data"]["zoneCreate"]


def does_camera_exist(
    portal_env: str,
    camera_uuid: str,
) -> bool:
    """Checks if camera exists for the specified organization and zone

    Args:
        portal_env (str): PROD or INTERNAL
        camera_uuid (str): name of the camera. Eg. americold/modesto/0001/cha

    Returns:
        bool: True if camera exists else False
    """
    with PerceptionPortalSession(portal_env) as perception_portal_session:
        query = "{cameras { uuid, organization { key } } }"
        query_results = perception_portal_session.session.post(
            f"{perception_portal_session.host}/graphql/",
            json={"query": query},
            headers=perception_portal_session.headers,
        )
        camera_info = json.loads(query_results.text)["data"]["cameras"]
        if not camera_info:
            logger.error("User does not have permission to view cameras")
            return False

        for camera in camera_info:
            if camera["uuid"] == camera_uuid:
                related_organization = camera["organization"]["key"]
                logger.info(
                    f"FOUND CAMERA, ORGANIZATION: {related_organization}, UUID: {camera_uuid}"
                )
                return True

        logger.info(f"CAMERA DOES NOT EXIST, UUID: {camera_uuid}")
        return False


def generate_camera_for_organization_zone(
    portal_env: str,
    organization_key: str,
    zone_key: str,
    camera_uuid: str,
    camera_name: str,
) -> dict:
    """If camera does not exist for the specified zone and organization, creat cameras.

    Args:
        portal_env (str): PROD or INTERNAL
        organization_key (str): organization uuid
        zone_key (str): zone uuid
        camera_uuid (str): camera name. Eg. ppg/cedar_falls/0001/cha
        camera_name (str): camera number Eg. 1, 2, 3

    Returns:
        dict: _description_
    """
    with PerceptionPortalSession(portal_env) as perception_portal_session:
        command = f'cameraCreate( \
            organizationKey: "{organization_key}", \
            zoneKey: "{zone_key}", \
            cameraUuid: "{camera_uuid}", \
            cameraName: "{camera_name}")'
        response = "camera {name, uuid}"
        mutation = f"mutation {{ {command} {{ {response} }} }}"
        mutation_results = perception_portal_session.session.post(
            f"{perception_portal_session.host}/graphql/",
            json={"query": mutation},
            headers=perception_portal_session.headers,
        )
        logger.info(f"MUTATION RESULTS: {mutation_results.text}")
        return json.loads(mutation_results.text)["data"]["cameraCreate"]
