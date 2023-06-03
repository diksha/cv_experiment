import { gql } from "@apollo/client";

export const GET_SITE_CAMERAS = gql`
  query GetSiteCameras($zoneId: String!) {
    zone(zoneId: $zoneId) {
      id
      cameras {
        edges {
          node {
            id
            name
            thumbnailUrl
            incidentTypes {
              id
              name
              key
              backgroundColor
            }
          }
        }
      }
    }
  }
`;
