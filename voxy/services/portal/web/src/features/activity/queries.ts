/*
 * Copyright 2020-2021 Voxel Labs, Inc.
 * All rights reserved.
 *
 * This document may not be reproduced, republished, distributed, transmitted,
 * displayed, broadcast or otherwise exploited in any manner without the express
 * prior written permission of Voxel Labs, Inc. The receipt or possession of this
 * document does not convey any rights to reproduce, disclose, or distribute its
 * contents, or to manufacture, use, or sell anything that it may describe, in
 * whole or in part.
 */
import { gql } from "@apollo/client";

export const GET_CURRENT_ZONE_ACTIVITY = gql`
  query GetCurrentZoneActivity($activityItemsPerFetch: Int!, $activityAfter: String) {
    currentUser {
      id
      zone {
        id
        recentComments(first: $activityItemsPerFetch, after: $activityAfter) {
          pageInfo {
            hasNextPage
            endCursor
          }
          edges {
            cursor
            node {
              id
              text
              createdAt
              incident {
                id
                uuid
                pk
                title
                priority
                status
                thumbnailUrl
                incidentType {
                  name
                }
              }
              owner {
                id
                fullName
                initials
                picture
              }
            }
          }
        }
      }
    }
  }
`;
