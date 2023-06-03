import { gql } from "@apollo/client";
export const GET_PRODUCTION_LINE_EVENTS = gql`
  query GetProductionLineEvents(
    $productionLineId: ID!
    $startTimestamp: DateTime!
    $endTimestamp: DateTime!
    $first: Int!
    $after: String
    $orderBy: String
  ) {
    productionLineDetails(productionLineId: $productionLineId) {
      id
      incidents(
        startTimestamp: $startTimestamp
        endTimestamp: $endTimestamp
        first: $first
        after: $after
        orderBy: $orderBy
      ) {
        pageInfo {
          hasNextPage
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
            priority
            status
            bookmarked
            highlighted
            alerted
            duration
            endTimestamp
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
    }
  }
`;

export const GET_PRODUCTION_LINES = gql`
  query GetProductionLines($startTimestamp: DateTime!, $endTimestamp: DateTime!) {
    currentUser {
      id
      site {
        id
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
