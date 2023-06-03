import { gql } from "@apollo/client";

export const USER_UPDATE = gql`
  mutation UserUpdate($userId: ID!, $firstName: String, $lastName: String, $roles: [String], $isActive: Boolean) {
    userUpdate(userId: $userId, firstName: $firstName, lastName: $lastName, roles: $roles, isActive: $isActive) {
      user {
        pk
        firstName
        lastName
        email
      }
    }
  }
`;

export const USER_RESEND_INVITATION = gql`
  mutation UserResendInvitation($invitationToken: String!) {
    userResendInvitation(invitationToken: $invitationToken) {
      status
    }
  }
`;

export const USER_INVITE = gql`
  mutation UserInvite($invitees: [InvitationInputSchema]!) {
    userInvite(invitees: $invitees) {
      status
    }
  }
`;

export const USER_REMOVE = gql`
  mutation UserRemove($userId: ID!) {
    userRemove(userId: $userId) {
      status
    }
  }
`;

export const USER_ROLE_UPDATE = gql`
  mutation UserRoleUpdate($userId: ID!, $roleId: ID!) {
    userRoleUpdate(userId: $userId, roleId: $roleId) {
      status
    }
  }
`;

export const USER_ZONES_UPDATE = gql`
  mutation UserZonesUpdate($userId: ID!, $zones: [ID!]!) {
    userZonesUpdate(userId: $userId, zones: $zones) {
      status
    }
  }
`;

export const USER_NAME_UPDATE = gql`
  mutation UserNameUpdate($userId: String!, $firstName: String, $lastName: String) {
    userNameUpdate(userId: $userId, firstName: $firstName, lastName: $lastName) {
      user {
        pk
        firstName
        lastName
        email
      }
    }
  }
`;

export const USER_MFA_UPDATE = gql`
  mutation UserMfaUpdate($userId: String!, $toggledMfaOn: Boolean!) {
    userMfaUpdate(userId: $userId, toggledMfaOn: $toggledMfaOn) {
      success
    }
  }
`;
