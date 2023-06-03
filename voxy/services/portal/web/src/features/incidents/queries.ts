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

export const GET_ALL_INCIDENT_TYPES = gql`
  query GetIncidentTypes {
    incidentTypes {
      key
      name
      backgroundColor
    }
  }
`;

export const GET_CURRENT_ORG_INCIDENT_TYPES = gql`
  query GetCurrentOrganizationIncidentTypes {
    currentUser {
      id
      organization {
        id
        incidentTypes {
          key
          name
          backgroundColor
        }
      }
    }
  }
`;

export const GET_INCIDENT_FEED = gql`
  query GetIncidentFeed(
    $fromUtc: DateTime
    $toUtc: DateTime
    $filters: [FilterInputType]
    $priorityFilter: [String]
    $statusFilter: [String]
    $incidentTypeFilter: [String]
    $cameraFilter: [String]
    $listFilter: [String]
    $assigneeFilter: [String]
    $experimentalFilter: Boolean
    $first: Int
    $after: String
  ) {
    incidentFeed(
      fromUtc: $fromUtc
      toUtc: $toUtc
      filters: $filters
      priorityFilter: $priorityFilter
      statusFilter: $statusFilter
      incidentTypeFilter: $incidentTypeFilter
      cameraFilter: $cameraFilter
      listFilter: $listFilter
      assigneeFilter: $assigneeFilter
      experimentalFilter: $experimentalFilter
      first: $first
      after: $after
    ) {
      edges {
        cursor
        node {
          id
          uuid
          pk
          title
          incidentType {
            id
            name
            key
          }
          timestamp
          endTimestamp
          duration
          priority
          status
          bookmarked
          highlighted
          alerted
          assignees {
            id
            initials
            fullName
          }
          camera {
            id
            name
            thumbnailUrl
          }
        }
      }
      pageInfo {
        startCursor
        endCursor
        hasNextPage
        hasPreviousPage
      }
    }
  }
`;

export const GET_INCIDENTS = gql`
  query GetIncidents($from: DateTime, $to: DateTime, $filters: [FilterInputType], $first: Int, $after: String) {
    incidentFeed(fromUtc: $from, toUtc: $to, filters: $filters, first: $first, after: $after) {
      edges {
        cursor
        node {
          id
          uuid
          pk
          title
          timestamp
          thumbnailUrl
          priority
          status
          bookmarked
          highlighted
        }
      }
      pageInfo {
        startCursor
        endCursor
        hasNextPage
        hasPreviousPage
      }
    }
  }
`;

export const GET_INCIDENT_DETAILS = gql`
  query GetIncidentDetails($incidentUuid: String!) {
    incidentDetails(incidentUuid: $incidentUuid) {
      id
      uuid
      pk
      title
      timestamp
      endTimestamp
      incidentType {
        id
        key
        name
        backgroundColor
      }
      priority
      status
      alerted
      bookmarked
      highlighted
      videoUrl
      annotationsUrl
      actorIds
      camera {
        id
        name
      }
      zone {
        id
        name
      }
      tags {
        label
        value
      }
      assignees {
        id
        fullName
      }
      organization {
        id
        name
      }
      duration
    }
  }
`;

export const GET_INCIDENT_FEEDBACK_QUEUE = gql`
  query GetIncidentFeedbackQueue($reviewQueueContext: ReviewQueueContext!) {
    incidentFeedbackQueue(reviewQueueContext: $reviewQueueContext) {
      id
      uuid
      pk
      title
      priority
      status
      videoUrl
      annotationsUrl
      actorIds
      organization {
        id
        name
      }
      zone {
        id
        name
      }
      dockerImageTag
      cameraUuid
      cameraConfig {
        doors
        drivingAreas
        actionableRegions
        intersections
        endOfAisles
        noPedestrianZones
        noObstructionRegions
      }
      incidentType {
        key
        description
      }
    }
  }
`;

export const GET_INCIDENT_FEEDBACK_FEED = gql`
  query GetIncidentFeedbackSummary(
    $internalFeedback: String
    $externalFeedback: String
    $hasComments: Boolean
    $incidentType: String
    $organizationId: String
    $siteId: String
    $fromUtc: DateTime
    $toUtc: DateTime
    $first: Int
    $after: String
  ) {
    incidentFeedbackSummary(
      internalFeedback: $internalFeedback
      externalFeedback: $externalFeedback
      hasComments: $hasComments
      incidentType: $incidentType
      organizationId: $organizationId
      siteId: $siteId
      fromUtc: $fromUtc
      toUtc: $toUtc
      first: $first
      after: $after
    ) {
      edges {
        cursor
        node {
          id
          uuid
          valid
          invalid
          unsure
          lastFeedbackSubmissionTimestamp
        }
      }
      pageInfo {
        startCursor
        endCursor
        hasNextPage
        hasPreviousPage
      }
    }
  }
`;

export const GET_INCIDENT_FEEDBACK_DETAIL = gql`
  query GetIncidentFeedbackDetails($incidentId: ID!) {
    incidentDetails(incidentId: $incidentId) {
      id
      pk
      title
      videoUrl
      annotationsUrl
      actorIds
      feedback {
        id
        feedbackText
        user {
          email
        }
      }
      cameraConfig {
        doors
        drivingAreas
        actionableRegions
        intersections
        endOfAisles
        noPedestrianZones
        noObstructionRegions
      }
    }
  }
`;

export const GET_INCIDENT_COMMENTS = gql`
  query GetComments($incidentId: ID!) {
    comments(incidentId: $incidentId) {
      id
      text
      note
      activityType
      createdAt
      owner {
        id
        fullName
        picture
      }
    }
  }
`;

export const GET_CURRENT_ZONE_INCIDENT_TYPE_STATS = gql`
  query GetCurrentZoneIncidentTypeStats(
    $startTimestamp: DateTime!
    $endTimestamp: DateTime!
    $filters: [FilterInputType]
  ) {
    currentUser {
      id
      zone {
        id
        incidentTypeStats(startTimestamp: $startTimestamp, endTimestamp: $endTimestamp, filters: $filters) {
          incidentType {
            key
            name
          }
          totalCount
        }
      }
    }
  }
`;

export const GET_CURRENT_ZONE_INCIDENT_FEED = gql`
  query GetCurrentZoneIncidentFeed(
    $startDate: Date
    $endDate: Date
    $filters: [FilterInputType]
    $timeBucketSizeHours: Int
    $after: String
  ) {
    currentUser {
      id
      zone {
        id
        timezone

        incidentFeed(
          startDate: $startDate
          endDate: $endDate
          filters: $filters
          timeBucketSizeHours: $timeBucketSizeHours
          after: $after
        ) {
          pageInfo {
            hasNextPage
            hasPreviousPage
            startCursor
            endCursor
          }
          edges {
            cursor
            node {
              __typename
              ... on DailyIncidentsFeedItem {
                key
                date
                timeBuckets {
                  key
                  title
                  startTimestamp
                  endTimestamp
                  incidentCount
                  incidentCounts {
                    count
                    incidentType {
                      key
                      name
                    }
                  }
                  latestIncidents {
                    id
                    uuid
                    title
                    timestamp
                    priority
                    status
                    bookmarked
                    highlighted
                    alerted
                    assignees {
                      id
                      initials
                      fullName
                    }
                    incidentType {
                      id
                      key
                      name
                    }
                    camera {
                      id
                      name
                      thumbnailUrl
                    }
                  }
                }
              }
              ... on EmptyRangeFeedItem {
                key
                startDate
                endDate
              }
            }
          }
        }
      }
    }
  }
`;

export const GET_DATA_PANEL_INCIDENTS = gql`
  query GetDataPanelIncidents(
    $startDate: Date
    $endDate: Date
    $startTimestamp: DateTime
    $endTimestamp: DateTime
    $filters: [FilterInputType]
    $first: Int!
    $after: String
    $groupBy: TimeBucketWidth!
  ) {
    currentUser {
      id
      site {
        id
        incidents(
          startDate: $startDate
          endDate: $endDate
          startTimestamp: $startTimestamp
          endTimestamp: $endTimestamp
          filters: $filters
          first: $first
          after: $after
        ) {
          pageInfo {
            hasNextPage
            hasPreviousPage
            startCursor
            endCursor
          }
          edges {
            cursor
            node {
              id
              uuid
              pk
              title
              incidentType {
                id
                name
                key
              }
              timestamp
              endTimestamp
              duration
              priority
              status
              bookmarked
              highlighted
              alerted
              assignees {
                id
                initials
                fullName
              }
              camera {
                id
                name
                thumbnailUrl
              }
            }
          }
        }
        incidentAnalytics {
          id
          incidentAggregateGroups(
            startTimestamp: $startTimestamp
            endTimestamp: $endTimestamp
            groupBy: $groupBy
            filters: $filters
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
      }
    }
  }
`;
