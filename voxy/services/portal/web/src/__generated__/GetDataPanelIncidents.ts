/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

import { FilterInputType, TimeBucketWidth, ApiIncidentPriorityChoices } from "./globalTypes";

// ====================================================
// GraphQL query operation: GetDataPanelIncidents
// ====================================================

export interface GetDataPanelIncidents_currentUser_site_incidents_pageInfo {
  __typename: "PageInfo";
  /**
   * When paginating forwards, are there more items?
   */
  hasNextPage: boolean;
  /**
   * When paginating backwards, are there more items?
   */
  hasPreviousPage: boolean;
  /**
   * When paginating backwards, the cursor to continue.
   */
  startCursor: string | null;
  /**
   * When paginating forwards, the cursor to continue.
   */
  endCursor: string | null;
}

export interface GetDataPanelIncidents_currentUser_site_incidents_edges_node_incidentType {
  __typename: "CameraIncidentType";
  id: string;
  name: string;
  key: string;
}

export interface GetDataPanelIncidents_currentUser_site_incidents_edges_node_assignees {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  initials: string | null;
  fullName: string | null;
}

export interface GetDataPanelIncidents_currentUser_site_incidents_edges_node_camera {
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

export interface GetDataPanelIncidents_currentUser_site_incidents_edges_node {
  __typename: "IncidentType";
  /**
   * The ID of the object
   */
  id: string;
  uuid: string;
  pk: number;
  title: string;
  incidentType: GetDataPanelIncidents_currentUser_site_incidents_edges_node_incidentType | null;
  timestamp: any;
  endTimestamp: any | null;
  duration: number | null;
  priority: ApiIncidentPriorityChoices;
  status: string | null;
  bookmarked: boolean;
  highlighted: boolean;
  alerted: boolean;
  assignees: (GetDataPanelIncidents_currentUser_site_incidents_edges_node_assignees | null)[];
  camera: GetDataPanelIncidents_currentUser_site_incidents_edges_node_camera | null;
}

export interface GetDataPanelIncidents_currentUser_site_incidents_edges {
  __typename: "IncidentTypeEdge";
  /**
   * A cursor for use in pagination
   */
  cursor: string;
  /**
   * The item at the end of the edge
   */
  node: GetDataPanelIncidents_currentUser_site_incidents_edges_node | null;
}

export interface GetDataPanelIncidents_currentUser_site_incidents {
  __typename: "IncidentTypeConnection";
  /**
   * Pagination data for this connection.
   */
  pageInfo: GetDataPanelIncidents_currentUser_site_incidents_pageInfo;
  /**
   * Contains the nodes in this connection.
   */
  edges: (GetDataPanelIncidents_currentUser_site_incidents_edges | null)[];
}

export interface GetDataPanelIncidents_currentUser_site_incidentAnalytics_incidentAggregateGroups_dimensions_incidentType {
  __typename: "OrganizationIncidentTypeType";
  id: string;
  key: string;
  name: string;
}

export interface GetDataPanelIncidents_currentUser_site_incidentAnalytics_incidentAggregateGroups_dimensions_camera {
  __typename: "CameraType";
  /**
   * The ID of the object
   */
  id: string;
  /**
   * User friendly name displayed throughout apps.
   */
  name: string;
  uuid: string;
}

export interface GetDataPanelIncidents_currentUser_site_incidentAnalytics_incidentAggregateGroups_dimensions {
  __typename: "IncidentAggregateDimensions";
  /**
   * Incident aggregate group datetime truncated to the appropriate date part based on the group_by property (e.g. hourly groups are truncated to the hour, daily groups truncated to the day, etc.). All values are localized to the parent zone's timezone.
   */
  datetime: any;
  incidentType: GetDataPanelIncidents_currentUser_site_incidentAnalytics_incidentAggregateGroups_dimensions_incidentType;
  camera: GetDataPanelIncidents_currentUser_site_incidentAnalytics_incidentAggregateGroups_dimensions_camera;
}

export interface GetDataPanelIncidents_currentUser_site_incidentAnalytics_incidentAggregateGroups_metrics {
  __typename: "IncidentAggregateMetrics";
  /**
   * Count of incidents in this group.
   */
  count: number;
}

export interface GetDataPanelIncidents_currentUser_site_incidentAnalytics_incidentAggregateGroups {
  __typename: "IncidentAggregateGroup";
  id: string;
  dimensions: GetDataPanelIncidents_currentUser_site_incidentAnalytics_incidentAggregateGroups_dimensions;
  metrics: GetDataPanelIncidents_currentUser_site_incidentAnalytics_incidentAggregateGroups_metrics;
}

export interface GetDataPanelIncidents_currentUser_site_incidentAnalytics {
  __typename: "SiteIncidentAnalytics";
  id: string;
  incidentAggregateGroups: GetDataPanelIncidents_currentUser_site_incidentAnalytics_incidentAggregateGroups[] | null;
}

export interface GetDataPanelIncidents_currentUser_site {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  incidents: GetDataPanelIncidents_currentUser_site_incidents | null;
  incidentAnalytics: GetDataPanelIncidents_currentUser_site_incidentAnalytics;
}

export interface GetDataPanelIncidents_currentUser {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  site: GetDataPanelIncidents_currentUser_site | null;
}

export interface GetDataPanelIncidents {
  currentUser: GetDataPanelIncidents_currentUser | null;
}

export interface GetDataPanelIncidentsVariables {
  startDate?: any | null;
  endDate?: any | null;
  startTimestamp?: any | null;
  endTimestamp?: any | null;
  filters?: (FilterInputType | null)[] | null;
  first: number;
  after?: string | null;
  groupBy: TimeBucketWidth;
}
