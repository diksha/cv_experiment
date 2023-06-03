/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetCurrentSiteAssignableUsers
// ====================================================

export interface GetCurrentSiteAssignableUsers_currentUser_site_assignableUsers_edges_node_sites {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  key: string;
  name: string;
}

export interface GetCurrentSiteAssignableUsers_currentUser_site_assignableUsers_edges_node {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  firstName: string;
  lastName: string;
  fullName: string | null;
  initials: string | null;
  email: string;
  isActive: boolean | null;
  sites: (GetCurrentSiteAssignableUsers_currentUser_site_assignableUsers_edges_node_sites | null)[] | null;
}

export interface GetCurrentSiteAssignableUsers_currentUser_site_assignableUsers_edges {
  __typename: "UserTypeEdge";
  /**
   * A cursor for use in pagination
   */
  cursor: string;
  /**
   * The item at the end of the edge
   */
  node: GetCurrentSiteAssignableUsers_currentUser_site_assignableUsers_edges_node | null;
}

export interface GetCurrentSiteAssignableUsers_currentUser_site_assignableUsers {
  __typename: "UserTypeConnection";
  /**
   * Contains the nodes in this connection.
   */
  edges: (GetCurrentSiteAssignableUsers_currentUser_site_assignableUsers_edges | null)[];
}

export interface GetCurrentSiteAssignableUsers_currentUser_site {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  assignableUsers: GetCurrentSiteAssignableUsers_currentUser_site_assignableUsers;
}

export interface GetCurrentSiteAssignableUsers_currentUser {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  site: GetCurrentSiteAssignableUsers_currentUser_site | null;
}

export interface GetCurrentSiteAssignableUsers {
  currentUser: GetCurrentSiteAssignableUsers_currentUser | null;
}
