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

export const GET_UNFILTERED_DASHBOARD_DATA = gql`
  query GetUnfilteredDashboardData(
    $highlightedEventsStartTimestamp: DateTime!
    $highlightedEventsEndTimestamp: DateTime!
  ) {
    currentUser {
      id
      stats {
        bookmarkTotalCount
        bookmarkResolvedCount
        bookmarkOpenCount
      }
      site {
        id
        latestActivityTimestamp
        allTimeIncidentStats: incidentStats {
          resolvedCount
        }
        incidentCategories {
          key
          name
          incidentTypes {
            key
            name
          }
        }
        highlightedEventsCount(
          startTimestamp: $highlightedEventsStartTimestamp
          endTimestamp: $highlightedEventsEndTimestamp
        )
      }
      tasksAssignedByStats {
        totalCount
        openCount
        resolvedCount
      }
      tasksAssignedToStats {
        totalCount
        openCount
        resolvedCount
      }
    }
  }
`;
export const GET_FILTERED_DASHBOARD_DATA = gql`
  query GetFilteredDashboardData($startTimestamp: DateTime!, $endTimestamp: DateTime!) {
    currentUser {
      id
      site {
        id
        filteredIncidentPriorityStats: incidentStats(startTimestamp: $startTimestamp, endTimestamp: $endTimestamp) {
          highPriorityCount
          mediumPriorityCount
          lowPriorityCount
        }
        filteredIncidentTypeStats: incidentTypeStats(startTimestamp: $startTimestamp, endTimestamp: $endTimestamp) {
          incidentType {
            key
            name
            category
            backgroundColor
          }
          totalCount
          maxTotalCount
        }
        filteredCameraStats: cameraStats(startTimestamp: $startTimestamp, endTimestamp: $endTimestamp) {
          camera {
            id
            uuid
            name
          }
          categoryStats {
            categoryKey
            totalCount
          }
        }
        filteredAssigneeStats: assigneeStats(startTimestamp: $startTimestamp, endTimestamp: $endTimestamp) {
          assignee {
            id
            fullName
          }
          openCount
          resolvedCount
          resolvedTimeAvgMinutes
        }
        productionLines {
          id
          uuid
          name
          camera {
            id
            name
          }
          status1hGroups(startTimestamp: $startTimestamp, endTimestamp: $endTimestamp, filters: []) {
            dimensions {
              datetime
            }
            metrics {
              uptimeDurationSeconds
              downtimeDurationSeconds
              unknownDurationSeconds
            }
          }
        }
      }
    }
  }
`;

export const GET_DASHBOARD_DATA = gql`
  query GetDashboardData(
    $startDate: Date!
    $endDate: Date!
    $startTimestamp: DateTime!
    $endTimestamp: DateTime!
    $groupBy: TimeBucketWidth!
  ) {
    currentUser {
      id
      stats {
        bookmarkTotalCount
      }
      site {
        id
        cameras {
          edges {
            node {
              id
              name
            }
          }
        }
        overallScore(startDate: $startDate, endDate: $endDate) {
          label
          value
        }
        eventScores(startDate: $startDate, endDate: $endDate) {
          label
          value
        }
        latestActivityTimestamp
        incidentCategories {
          key
          name
          incidentTypes {
            key
            name
            backgroundColor
          }
        }
        assigneeStats(startDate: $startDate, endDate: $endDate) {
          assignee {
            id
            fullName
          }
          openCount
          resolvedCount
        }
        incidentAnalytics {
          id
          incidentAggregateGroups(
            startDate: $startDate
            endDate: $endDate
            startTimestamp: $startTimestamp
            endTimestamp: $endTimestamp
            groupBy: $groupBy
          ) {
            id
            dimensions {
              datetime
              incidentType {
                id
                key
                name
              }
              camera {
                id
                name
                uuid
              }
            }
            metrics {
              count
            }
          }
        }
        productionLines {
          id
          uuid
          name
          camera {
            id
            name
          }
          status1hGroups(startTimestamp: $startTimestamp, endTimestamp: $endTimestamp, filters: []) {
            dimensions {
              datetime
            }
            metrics {
              uptimeDurationSeconds
              downtimeDurationSeconds
              unknownDurationSeconds
            }
          }
        }
      }
      tasksAssignedByStats {
        totalCount
        openCount
        resolvedCount
      }
      tasksAssignedToStats {
        totalCount
        openCount
        resolvedCount
      }
    }
  }
`;
