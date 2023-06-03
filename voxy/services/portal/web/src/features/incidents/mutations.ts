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

export const ASSIGN_INCIDENT = gql`
  mutation AssignIncident($incidentId: ID!, $assigneeIds: [ID], $note: String!) {
    assignIncident(assignData: { incidentId: $incidentId, assigneeIds: $assigneeIds, note: $note }) {
      users {
        id
        firstName
        lastName
      }
    }
  }
`;

export const UNASSIGN_INCIDENT = gql`
  mutation UnassignIncident($incidentId: ID!, $assigneeId: ID!) {
    unassignIncident(incidentId: $incidentId, assigneeId: $assigneeId) {
      assign {
        id
      }
    }
  }
`;

export const CREATE_INCIDENT_FEEDBACK = gql`
  mutation CreateIncidentFeedback(
    $incidentId: ID!
    $feedbackType: String!
    $feedbackValue: String!
    $feedbackText: String!
    $elapsedMillisecondsBetweenReviews: Int
    $incidentServedTimestampSeconds: Int
  ) {
    createIncidentFeedback(
      incidentId: $incidentId
      feedbackType: $feedbackType
      feedbackValue: $feedbackValue
      feedbackText: $feedbackText
      elapsedMillisecondsBetweenReviews: $elapsedMillisecondsBetweenReviews
      incidentServedTimestampSeconds: $incidentServedTimestampSeconds
    ) {
      incidentFeedback {
        id
        feedbackType
        feedbackValue
        feedbackText
        updatedAt
      }

      userErrors {
        message
        code
      }
    }
  }
`;

export const ADD_BOOKMARK = gql`
  mutation CurrentUserAddBookmark($incidentId: ID!) {
    currentUserAddBookmark(incidentId: $incidentId) {
      incident {
        id
        bookmarked
      }
    }
  }
`;

export const REMOVE_BOOKMARK = gql`
  mutation CurrentUserRemoveBookmark($incidentId: ID!) {
    currentUserRemoveBookmark(incidentId: $incidentId) {
      incident {
        id
        bookmarked
      }
    }
  }
`;

export const INCIDENT_HIGHLIGHT = gql`
  mutation IncidentHighlight($incidentId: ID!) {
    incidentHighlight(incidentId: $incidentId) {
      incident {
        id
        highlighted
      }
    }
  }
`;

export const INCIDENT_UNDO_HIGHLIGHT = gql`
  mutation IncidentUndoHighlight($incidentId: ID!) {
    incidentUndoHighlight(incidentId: $incidentId) {
      incident {
        id
        highlighted
      }
    }
  }
`;

export const RESOLVE_INCIDENT = gql`
  mutation IncidentResolve($incidentId: ID!) {
    incidentResolve(incidentId: $incidentId) {
      incident {
        id
        status
      }
    }
  }
`;

export const REOPEN_INCIDENT = gql`
  mutation IncidentReopen($incidentId: ID!) {
    incidentReopen(incidentId: $incidentId) {
      incident {
        id
        status
      }
    }
  }
`;

export const CREATE_COMMENT = gql`
  mutation CreateComment($incidentId: ID!, $text: String!) {
    createComment(incidentId: $incidentId, text: $text) {
      comment {
        id
        text
        createdAt
        activityType
        note
        owner {
          id
          firstName
          lastName
          picture
        }
      }
    }
  }
`;

export const EXPORT_VIDEO = gql`
  mutation IncidentExportVideo($incidentId: ID!, $labeled: Boolean!) {
    incidentExportVideo(incidentId: $incidentId, labeled: $labeled) {
      downloadUrl
    }
  }
`;

export const CREATE_SHARE_LINK = gql`
  mutation IncidentCreateShareLink($incidentId: ID!) {
    incidentCreateShareLink(incidentId: $incidentId) {
      shareLink
    }
  }
`;
