/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

import { ApiIncidentPriorityChoices } from "./globalTypes";

// ====================================================
// GraphQL query operation: GetProductionLineEvents
// ====================================================

export interface GetProductionLineEvents_productionLineDetails_incidents_pageInfo {
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

export interface GetProductionLineEvents_productionLineDetails_incidents_edges_node_incidentType {
  __typename: "CameraIncidentType";
  id: string;
  name: string;
  key: string;
}

export interface GetProductionLineEvents_productionLineDetails_incidents_edges_node_assignees {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  initials: string | null;
  fullName: string | null;
}

export interface GetProductionLineEvents_productionLineDetails_incidents_edges_node_camera {
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

export interface GetProductionLineEvents_productionLineDetails_incidents_edges_node {
  __typename: "IncidentType";
  /**
   * The ID of the object
   */
  id: string;
  uuid: string;
  pk: number;
  title: string;
  incidentType: GetProductionLineEvents_productionLineDetails_incidents_edges_node_incidentType | null;
  timestamp: any;
  priority: ApiIncidentPriorityChoices;
  status: string | null;
  bookmarked: boolean;
  highlighted: boolean;
  alerted: boolean;
  duration: number | null;
  endTimestamp: any | null;
  assignees: (GetProductionLineEvents_productionLineDetails_incidents_edges_node_assignees | null)[];
  camera: GetProductionLineEvents_productionLineDetails_incidents_edges_node_camera | null;
}

export interface GetProductionLineEvents_productionLineDetails_incidents_edges {
  __typename: "IncidentTypeEdge";
  /**
   * A cursor for use in pagination
   */
  cursor: string;
  /**
   * The item at the end of the edge
   */
  node: GetProductionLineEvents_productionLineDetails_incidents_edges_node | null;
}

export interface GetProductionLineEvents_productionLineDetails_incidents {
  __typename: "IncidentTypeConnection";
  /**
   * Pagination data for this connection.
   */
  pageInfo: GetProductionLineEvents_productionLineDetails_incidents_pageInfo;
  /**
   * Contains the nodes in this connection.
   */
  edges: (GetProductionLineEvents_productionLineDetails_incidents_edges | null)[];
}

export interface GetProductionLineEvents_productionLineDetails {
  __typename: "ProductionLine";
  id: string;
  incidents: GetProductionLineEvents_productionLineDetails_incidents | null;
}

export interface GetProductionLineEvents {
  productionLineDetails: GetProductionLineEvents_productionLineDetails | null;
}

export interface GetProductionLineEventsVariables {
  productionLineId: string;
  startTimestamp: any;
  endTimestamp: any;
  first: number;
  after?: string | null;
  orderBy?: string | null;
}
