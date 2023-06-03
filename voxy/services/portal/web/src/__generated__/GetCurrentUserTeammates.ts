/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetCurrentUserTeammates
// ====================================================

export interface GetCurrentUserTeammates_currentUser_invitedUsers_user {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  email: string;
}

export interface GetCurrentUserTeammates_currentUser_invitedUsers_role {
  __typename: "Role";
  id: string;
  name: string;
  key: string;
}

export interface GetCurrentUserTeammates_currentUser_invitedUsers_sites {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  key: string;
  name: string;
}

export interface GetCurrentUserTeammates_currentUser_invitedUsers {
  __typename: "InvitedUserType";
  user: GetCurrentUserTeammates_currentUser_invitedUsers_user | null;
  role: GetCurrentUserTeammates_currentUser_invitedUsers_role;
  sites: (GetCurrentUserTeammates_currentUser_invitedUsers_sites | null)[] | null;
  expired: boolean;
  createdAt: any;
  token: string;
}

export interface GetCurrentUserTeammates_currentUser_teammates_edges_node_roles {
  __typename: "Role";
  id: string;
  name: string;
  key: string;
}

export interface GetCurrentUserTeammates_currentUser_teammates_edges_node_sites {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  key: string;
  name: string;
}

export interface GetCurrentUserTeammates_currentUser_teammates_edges_node {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  firstName: string;
  lastName: string;
  fullName: string | null;
  email: string;
  isActive: boolean | null;
  roles: GetCurrentUserTeammates_currentUser_teammates_edges_node_roles[] | null;
  sites: (GetCurrentUserTeammates_currentUser_teammates_edges_node_sites | null)[] | null;
}

export interface GetCurrentUserTeammates_currentUser_teammates_edges {
  __typename: "UserTypeEdge";
  /**
   * A cursor for use in pagination
   */
  cursor: string;
  /**
   * The item at the end of the edge
   */
  node: GetCurrentUserTeammates_currentUser_teammates_edges_node | null;
}

export interface GetCurrentUserTeammates_currentUser_teammates {
  __typename: "UserTypeConnection";
  /**
   * Contains the nodes in this connection.
   */
  edges: (GetCurrentUserTeammates_currentUser_teammates_edges | null)[];
}

export interface GetCurrentUserTeammates_currentUser {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  invitedUsers: GetCurrentUserTeammates_currentUser_invitedUsers[] | null;
  teammates: GetCurrentUserTeammates_currentUser_teammates | null;
}

export interface GetCurrentUserTeammates {
  currentUser: GetCurrentUserTeammates_currentUser | null;
}
