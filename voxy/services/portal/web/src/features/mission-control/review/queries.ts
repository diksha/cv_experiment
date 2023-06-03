import { gql } from "@apollo/client";

export const GET_ALL_ORGANIZATIONS_WITH_SITES = gql`
  query GetAllOrganizationsWithSites {
    organizations {
      edges {
        cursor
        node {
          id
          pk
          name
          sites {
            id
            name
          }
        }
      }
    }
  }
`;
