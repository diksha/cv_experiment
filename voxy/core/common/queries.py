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

FILTERED_RAW_INCIDENTS_QUERY = """
query GetFilteredIncidents($fromUtc: DateTime, $toUtc: DateTime, $incidentTypeFilter: String, $organizationKey: [String], $zoneKey: [String], $cameraUuid: [String], $feedbackType: String, $first: Int, $after: String) {
    integrations {
        filteredRawIncidents(
            fromUtc: $fromUtc
            toUtc: $toUtc
            incidentTypeFilter: $incidentTypeFilter
            organizationKey: $organizationKey
            zoneKey: $zoneKey
            cameraUuid: $cameraUuid
            feedbackType: $feedbackType
            first: $first
            after: $after
        ) {
            pageInfo {
                hasNextPage
                endCursor
            }
            edges {
                cursor
                node {
                    uuid
                    data
                    cameraUuid
                    experimental
                    organization {
                        key
                    }
                    zone {
                        key
                    }
                    incidentType {
                        key
                    }
                }
            }
        }
    }
}
"""

ACTIVE_CAMERAS = """
    {
    cameras{
        uuid
        zone {
            isActive
            }
        }
    }
"""

INCIDENT_DETAILS = """
query GetIncidentDetails($incidentUuid: String) {
    incidentDetails(incidentUuid: $incidentUuid){
       uuid
        data
        createdAt
        cameraUuid
        incidentType {
          key
        }
        experimental
        organization {
          key
        }
        zone {
          key
        }
        validFeedbackCount
        invalidFeedbackCount
        unsureFeedbackCount
      }
}
"""
