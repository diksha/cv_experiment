import { gql } from "@apollo/client";

export const GET_ALL_ORGANIZATIONS = gql`
  query GetAllOrganizations {
    organizations {
      edges {
        cursor
        node {
          id
          pk
          name
        }
      }
    }
  }
`;

export const GET_CAMERAS = gql`
  query GetCameras {
    cameras {
      uuid
      name
    }
  }
`;

export const GET_CURRENT_SITE_ASSIGNABLE_USERS = gql`
  query GetCurrentSiteAssignableUsers {
    currentUser {
      id
      site {
        id
        assignableUsers {
          edges {
            cursor
            node {
              id
              firstName
              lastName
              fullName
              initials
              email
              isActive
              sites {
                id
                key
                name
              }
            }
          }
        }
      }
    }
  }
`;

export const GET_CURRENT_USER_TEAMMATES = gql`
  query GetCurrentUserTeammates {
    currentUser {
      id
      invitedUsers {
        user {
          id
          email
        }
        role {
          id
          name
          key
        }
        sites {
          id
          key
          name
        }
        expired
        createdAt
        token
      }
      teammates {
        edges {
          cursor
          node {
            id
            firstName
            lastName
            fullName
            email
            isActive
            roles {
              id
              name
              key
            }
            sites {
              id
              key
              name
            }
          }
        }
      }
    }
  }
`;

export const GET_CURRENT_SITE_DATA = gql`
  query GetCurrentSiteData {
    currentUser {
      id
      site {
        id
        key
        name
        isActive
      }
      sites {
        id
        key
        name
        isActive
      }
      organization {
        id
        name
      }
    }
  }
`;
