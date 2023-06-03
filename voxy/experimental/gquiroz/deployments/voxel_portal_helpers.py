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

import json

from core.utils.perception_portal_graphql_session import (
    PerceptionPortalSession,
)


def does_organization_exist(
    portal_env: str,
    organization_key: str,
) -> bool:
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
        print(f"MUTATION RESULTS: {mutation_results.text}")
        return json.loads(mutation_results.text)["data"]["organizationCreate"]


def does_zone_exist_for_organization(
    portal_env: str,
    organization_key: str,
    zone_key: str,
) -> bool:
    with PerceptionPortalSession(portal_env) as perception_portal_session:
        query = f'{{organization(organizationKey: "{organization_key}") {{ sites {{ key }} }} }}'
        query_results = perception_portal_session.session.post(
            f"{perception_portal_session.host}/graphql/",
            json={"query": query},
            headers=perception_portal_session.headers,
        )
        org_info = json.loads(query_results.text)["data"]["organization"]
        if not org_info:
            print(f"ORGANIZATION DOES NOT EXIST: {organization_key}")
            return False

        print(org_info)
        zones = org_info["sites"]
        for zone in zones:
            if zone["key"] == zone_key:
                print(
                    f"FOUND ZONE, ORGANIZATION: {organization_key}, ZONE: {zone_key}"
                )
                return True

        print(
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
        print(f"MUTATION RESULTS: {mutation_results.text}")
        return json.loads(mutation_results.text)["data"]["zoneCreate"]


def does_camera_exist(
    portal_env: str,
    camera_uuid: str,
) -> bool:
    with PerceptionPortalSession(portal_env) as perception_portal_session:
        query = "{cameras { uuid, organization { key } } }"
        query_results = perception_portal_session.session.post(
            f"{perception_portal_session.host}/graphql/",
            json={"query": query},
            headers=perception_portal_session.headers,
        )
        camera_info = json.loads(query_results.text)["data"]["cameras"]
        if not camera_info:
            print("User does not have zone to view cameras")
            return False

        for camera in camera_info:
            if camera["uuid"] == camera_uuid:
                related_organization = camera["organization"]["key"]
                print(
                    f"FOUND CAMERA, ORGANIZATION: {related_organization}, UUID: {camera_uuid}"
                )
                return True

        print(f"CAMERA DOES NOT EXIST, UUID: {camera_uuid}")
        return False


def generate_camera_for_organization_zone(
    portal_env: str,
    organization_key: str,
    zone_key: str,
    camera_uuid: str,
    camera_name: str,
) -> dict:
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
        print(f"MUTATION RESULTS: {mutation_results.text}")
        return json.loads(mutation_results.text)["data"]["cameraCreate"]
