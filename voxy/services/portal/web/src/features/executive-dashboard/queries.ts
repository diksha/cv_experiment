import { gql } from "@apollo/client";

export const GET_EXECUTIVE_DASHBOARD_DATA = gql`
  query GetExecutiveDashboardData(
    $startDate: Date!
    $endDate: Date!
    $groupBy: TimeBucketWidth!
    $filters: [FilterInputType]
  ) {
    currentUser {
      id
      fullName
      organization {
        id
        name
        incidentTypes {
          key
          name
          backgroundColor
        }
        overallScore(startDate: $startDate, endDate: $endDate) {
          label
          value
        }
        eventScores(startDate: $startDate, endDate: $endDate) {
          label
          value
        }
        sessionCount(startDate: $startDate, endDate: $endDate) {
          users {
            user {
              id
              email
              fullName
              picture
              sites {
                name
              }
            }
            value
          }
          sites {
            site {
              id
              name
              isActive
            }
            value
          }
        }
        sites {
          id
          key
          name
          timezone
          isActive
          overallScore(startDate: $startDate, endDate: $endDate) {
            label
            value
          }
          eventScores(startDate: $startDate, endDate: $endDate) {
            label
            value
          }
          incidentAnalytics {
            incidentAggregateGroups(startDate: $startDate, endDate: $endDate, groupBy: $groupBy, filters: $filters) {
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
  }
`;

export const GET_EXECUTIVE_DASHBOARD_SITE_DATA_PANEL = gql`
  query GetExecutiveDashboardSiteDataPanel(
    $zoneId: String
    $startDate: Date!
    $endDate: Date!
    $groupBy: TimeBucketWidth!
  ) {
    zone(zoneId: $zoneId) {
      id
      key
      name
      timezone
      isActive
      incidentTypes {
        key
        name
        backgroundColor
      }
      overallScore(startDate: $startDate, endDate: $endDate) {
        label
        value
      }
      eventScores(startDate: $startDate, endDate: $endDate) {
        label
        value
      }
      sessionCount(startDate: $startDate, endDate: $endDate) {
        users {
          user {
            id
            email
            fullName
            picture
            sites {
              name
            }
          }
          value
        }
      }
      incidentAnalytics {
        incidentAggregateGroups(startDate: $startDate, endDate: $endDate, groupBy: $groupBy) {
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
`;

export const GET_EXECUTIVE_DASHBOARD_ACTIVE_EMPLOYEES = gql`
  query GetExecutiveDashboardActiveEmployees($startDate: Date!, $endDate: Date!) {
    currentUser {
      id
      organization {
        id
        name
        sessionCount(startDate: $startDate, endDate: $endDate) {
          users {
            user {
              id
              email
              fullName
              picture
              sites {
                name
              }
            }
            value
          }
        }
      }
    }
  }
`;
