import { gql } from "@apollo/client";

export const GET_ALL_ORGANIZATIONS_AND_SITES = gql`
  query GetAllOrganizationsAndSites {
    organizations {
      edges {
        cursor
        node {
          id
          pk
          name
          isSandbox
          sites {
            id
            name
            isActive
          }
        }
      }
    }
  }
`;

export const INCIDENT_CREATE_SCENARIO = gql`
  mutation IncidentCreateScenario($incidentId: ID!, $scenarioType: ScenarioType!) {
    incidentCreateScenario(incidentId: $incidentId, scenarioType: $scenarioType) {
      incident {
        id
        pk
        data
      }
    }
  }
`;
