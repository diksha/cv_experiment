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

STAGING_CAMERA_CREATE_QUERY = """
    mutation createCamera(
        $cameraUuid:String!,
        $cameraName:String!,
        $organizationKey:String!,
        $zoneKey:String!
    ) {
            cameraCreate(
                cameraUuid: $cameraUuid,
                cameraName: $cameraName,
                organizationKey: $organizationKey,
                zoneKey: $zoneKey
            ) {
                camera {
                    uuid
                }
            }
}
"""


STAGING_ORGANIZATION_CREATE_QUERY = """
mutation($organizationKey:String!, $organizationName:String!){
    organizationCreate(organizationKey:$organizationKey, organizationName:$organizationName) {
        organization{
            key
        }
    }
}
"""

STAGING_ZONE_CREATE_QUERY = """
mutation($zoneKey: String!, $zoneName: String!, $zoneType: String!, $organizationKey: String!){
  zoneCreate(zoneKey:$zoneKey, zoneName:$zoneName, zoneType:$zoneType, organizationKey:$organizationKey) {
    zone{
      key
    }
  }
}
"""
