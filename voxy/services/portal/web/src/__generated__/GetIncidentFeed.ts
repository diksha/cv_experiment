/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

import { FilterInputType, ApiIncidentPriorityChoices } from "./globalTypes";

// ====================================================
// GraphQL query operation: GetIncidentFeed
// ====================================================

export interface GetIncidentFeed_incidentFeed_edges_node_incidentType {
  __typename: "CameraIncidentType";
  id: string;
  name: string;
  key: string;
}

export interface GetIncidentFeed_incidentFeed_edges_node_assignees {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  initials: string | null;
  fullName: string | null;
}

export interface GetIncidentFeed_incidentFeed_edges_node_camera {
  __typename: "CameraType";
  /**
   * The ID of the object
   */
  id: string;
  /**
   * User friendly name displayed throughout apps.
   */
  name: string;
  thumbnailUrl: string | null;
}

export interface GetIncidentFeed_incidentFeed_edges_node {
  __typename: "IncidentType";
  /**
   * The ID of the object
   */
  id: string;
  uuid: string;
  pk: number;
  title: string;
  incidentType: GetIncidentFeed_incidentFeed_edges_node_incidentType | null;
  timestamp: any;
  endTimestamp: any | null;
  duration: number | null;
  priority: ApiIncidentPriorityChoices;
  status: string | null;
  bookmarked: boolean;
  highlighted: boolean;
  alerted: boolean;
  assignees: (GetIncidentFeed_incidentFeed_edges_node_assignees | null)[];
  camera: GetIncidentFeed_incidentFeed_edges_node_camera | null;
}

export interface GetIncidentFeed_incidentFeed_edges {
  __typename: "IncidentTypeEdge";
  /**
   * A cursor for use in pagination
   */
  cursor: string;
  /**
   * The item at the end of the edge
   */
  node: GetIncidentFeed_incidentFeed_edges_node | null;
}

export interface GetIncidentFeed_incidentFeed_pageInfo {
  __typename: "PageInfo";
  /**
   * When paginating backwards, the cursor to continue.
   */
  startCursor: string | null;
  /**
   * When paginating forwards, the cursor to continue.
   */
  endCursor: string | null;
  /**
   * When paginating forwards, are there more items?
   */
  hasNextPage: boolean;
  /**
   * When paginating backwards, are there more items?
   */
  hasPreviousPage: boolean;
}

export interface GetIncidentFeed_incidentFeed {
  __typename: "IncidentTypeConnection";
  /**
   * Contains the nodes in this connection.
   */
  edges: (GetIncidentFeed_incidentFeed_edges | null)[];
  /**
   * Pagination data for this connection.
   */
  pageInfo: GetIncidentFeed_incidentFeed_pageInfo;
}

export interface GetIncidentFeed {
  incidentFeed: GetIncidentFeed_incidentFeed | null;
}

export interface GetIncidentFeedVariables {
  fromUtc?: any | null;
  toUtc?: any | null;
  filters?: (FilterInputType | null)[] | null;
  priorityFilter?: (string | null)[] | null;
  statusFilter?: (string | null)[] | null;
  incidentTypeFilter?: (string | null)[] | null;
  cameraFilter?: (string | null)[] | null;
  listFilter?: (string | null)[] | null;
  assigneeFilter?: (string | null)[] | null;
  experimentalFilter?: boolean | null;
  first?: number | null;
  after?: string | null;
}
