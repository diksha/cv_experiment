/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetFilterBarData
// ====================================================

export interface GetFilterBarData_currentUser_organization_incidentTypes {
  __typename: "OrganizationIncidentTypeType";
  key: string;
  name: string;
}

export interface GetFilterBarData_currentUser_organization {
  __typename: "OrganizationType";
  /**
   * The ID of the object
   */
  id: string;
  incidentTypes: GetFilterBarData_currentUser_organization_incidentTypes[];
}

export interface GetFilterBarData_currentUser_site_cameras_edges_node {
  __typename: "CameraType";
  /**
   * The ID of the object
   */
  id: string;
  /**
   * User friendly name displayed throughout apps.
   */
  name: string;
}

export interface GetFilterBarData_currentUser_site_cameras_edges {
  __typename: "CameraTypeEdge";
  /**
   * The item at the end of the edge
   */
  node: GetFilterBarData_currentUser_site_cameras_edges_node | null;
}

export interface GetFilterBarData_currentUser_site_cameras {
  __typename: "CameraTypeConnection";
  /**
   * Contains the nodes in this connection.
   */
  edges: (GetFilterBarData_currentUser_site_cameras_edges | null)[];
}

export interface GetFilterBarData_currentUser_site_assignableUsers_edges_node {
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
}

export interface GetFilterBarData_currentUser_site_assignableUsers_edges {
  __typename: "UserTypeEdge";
  /**
   * A cursor for use in pagination
   */
  cursor: string;
  /**
   * The item at the end of the edge
   */
  node: GetFilterBarData_currentUser_site_assignableUsers_edges_node | null;
}

export interface GetFilterBarData_currentUser_site_assignableUsers {
  __typename: "UserTypeConnection";
  /**
   * Contains the nodes in this connection.
   */
  edges: (GetFilterBarData_currentUser_site_assignableUsers_edges | null)[];
}

export interface GetFilterBarData_currentUser_site {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  cameras: GetFilterBarData_currentUser_site_cameras | null;
  assignableUsers: GetFilterBarData_currentUser_site_assignableUsers;
}

export interface GetFilterBarData_currentUser {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  organization: GetFilterBarData_currentUser_organization | null;
  site: GetFilterBarData_currentUser_site | null;
}

export interface GetFilterBarData {
  currentUser: GetFilterBarData_currentUser | null;
}
