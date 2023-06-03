/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

import { FilterInputType, ApiIncidentPriorityChoices } from "./globalTypes";

// ====================================================
// GraphQL query operation: GetIncidents
// ====================================================

export interface GetIncidents_incidentFeed_edges_node {
  __typename: "IncidentType";
  /**
   * The ID of the object
   */
  id: string;
  uuid: string;
  pk: number;
  title: string;
  timestamp: any;
  thumbnailUrl: string;
  priority: ApiIncidentPriorityChoices;
  status: string | null;
  bookmarked: boolean;
  highlighted: boolean;
}

export interface GetIncidents_incidentFeed_edges {
  __typename: "IncidentTypeEdge";
  /**
   * A cursor for use in pagination
   */
  cursor: string;
  /**
   * The item at the end of the edge
   */
  node: GetIncidents_incidentFeed_edges_node | null;
}

export interface GetIncidents_incidentFeed_pageInfo {
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

export interface GetIncidents_incidentFeed {
  __typename: "IncidentTypeConnection";
  /**
   * Contains the nodes in this connection.
   */
  edges: (GetIncidents_incidentFeed_edges | null)[];
  /**
   * Pagination data for this connection.
   */
  pageInfo: GetIncidents_incidentFeed_pageInfo;
}

export interface GetIncidents {
  incidentFeed: GetIncidents_incidentFeed | null;
}

export interface GetIncidentsVariables {
  from?: any | null;
  to?: any | null;
  filters?: (FilterInputType | null)[] | null;
  first?: number | null;
  after?: string | null;
}
