/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

import { ApiIncidentPriorityChoices } from "./globalTypes";

// ====================================================
// GraphQL query operation: GetCurrentZoneActivity
// ====================================================

export interface GetCurrentZoneActivity_currentUser_zone_recentComments_pageInfo {
  __typename: "PageInfo";
  /**
   * When paginating forwards, are there more items?
   */
  hasNextPage: boolean;
  /**
   * When paginating forwards, the cursor to continue.
   */
  endCursor: string | null;
}

export interface GetCurrentZoneActivity_currentUser_zone_recentComments_edges_node_incident_incidentType {
  __typename: "CameraIncidentType";
  name: string;
}

export interface GetCurrentZoneActivity_currentUser_zone_recentComments_edges_node_incident {
  __typename: "IncidentType";
  /**
   * The ID of the object
   */
  id: string;
  uuid: string;
  pk: number;
  title: string;
  priority: ApiIncidentPriorityChoices;
  status: string | null;
  thumbnailUrl: string;
  incidentType: GetCurrentZoneActivity_currentUser_zone_recentComments_edges_node_incident_incidentType | null;
}

export interface GetCurrentZoneActivity_currentUser_zone_recentComments_edges_node_owner {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  fullName: string | null;
  initials: string | null;
  picture: string | null;
}

export interface GetCurrentZoneActivity_currentUser_zone_recentComments_edges_node {
  __typename: "CommentType";
  /**
   * The ID of the object
   */
  id: string;
  text: string;
  createdAt: any;
  incident: GetCurrentZoneActivity_currentUser_zone_recentComments_edges_node_incident | null;
  owner: GetCurrentZoneActivity_currentUser_zone_recentComments_edges_node_owner | null;
}

export interface GetCurrentZoneActivity_currentUser_zone_recentComments_edges {
  __typename: "CommentTypeEdge";
  /**
   * A cursor for use in pagination
   */
  cursor: string;
  /**
   * The item at the end of the edge
   */
  node: GetCurrentZoneActivity_currentUser_zone_recentComments_edges_node | null;
}

export interface GetCurrentZoneActivity_currentUser_zone_recentComments {
  __typename: "CommentTypeConnection";
  /**
   * Pagination data for this connection.
   */
  pageInfo: GetCurrentZoneActivity_currentUser_zone_recentComments_pageInfo;
  /**
   * Contains the nodes in this connection.
   */
  edges: (GetCurrentZoneActivity_currentUser_zone_recentComments_edges | null)[];
}

export interface GetCurrentZoneActivity_currentUser_zone {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  recentComments: GetCurrentZoneActivity_currentUser_zone_recentComments | null;
}

export interface GetCurrentZoneActivity_currentUser {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  zone: GetCurrentZoneActivity_currentUser_zone | null;
}

export interface GetCurrentZoneActivity {
  currentUser: GetCurrentZoneActivity_currentUser | null;
}

export interface GetCurrentZoneActivityVariables {
  activityItemsPerFetch: number;
  activityAfter?: string | null;
}
