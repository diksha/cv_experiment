/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

import { TimeBucketWidth } from "./globalTypes";

// ====================================================
// GraphQL query operation: GetDashboardData
// ====================================================

export interface GetDashboardData_currentUser_stats {
  __typename: "UserStats";
  bookmarkTotalCount: number | null;
}

export interface GetDashboardData_currentUser_site_cameras_edges_node {
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

export interface GetDashboardData_currentUser_site_cameras_edges {
  __typename: "CameraTypeEdge";
  /**
   * The item at the end of the edge
   */
  node: GetDashboardData_currentUser_site_cameras_edges_node | null;
}

export interface GetDashboardData_currentUser_site_cameras {
  __typename: "CameraTypeConnection";
  /**
   * Contains the nodes in this connection.
   */
  edges: (GetDashboardData_currentUser_site_cameras_edges | null)[];
}

export interface GetDashboardData_currentUser_site_overallScore {
  __typename: "Score";
  label: string;
  value: number;
}

export interface GetDashboardData_currentUser_site_eventScores {
  __typename: "Score";
  label: string;
  value: number;
}

export interface GetDashboardData_currentUser_site_incidentCategories_incidentTypes {
  __typename: "IncidentTypeType";
  key: string;
  name: string;
  backgroundColor: string | null;
}

export interface GetDashboardData_currentUser_site_incidentCategories {
  __typename: "IncidentCategory";
  key: string | null;
  name: string | null;
  incidentTypes: (GetDashboardData_currentUser_site_incidentCategories_incidentTypes | null)[] | null;
}

export interface GetDashboardData_currentUser_site_assigneeStats_assignee {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  fullName: string | null;
}

export interface GetDashboardData_currentUser_site_assigneeStats {
  __typename: "AssigneeStats";
  assignee: GetDashboardData_currentUser_site_assigneeStats_assignee | null;
  openCount: number | null;
  resolvedCount: number | null;
}

export interface GetDashboardData_currentUser_site_incidentAnalytics_incidentAggregateGroups_dimensions_incidentType {
  __typename: "OrganizationIncidentTypeType";
  id: string;
  key: string;
  name: string;
}

export interface GetDashboardData_currentUser_site_incidentAnalytics_incidentAggregateGroups_dimensions_camera {
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

export interface GetDashboardData_currentUser_site_incidentAnalytics_incidentAggregateGroups_dimensions {
  __typename: "IncidentAggregateDimensions";
  /**
   * Incident aggregate group datetime truncated to the appropriate date part based on the group_by property (e.g. hourly groups are truncated to the hour, daily groups truncated to the day, etc.). All values are localized to the parent zone's timezone.
   */
  datetime: any;
  incidentType: GetDashboardData_currentUser_site_incidentAnalytics_incidentAggregateGroups_dimensions_incidentType;
  camera: GetDashboardData_currentUser_site_incidentAnalytics_incidentAggregateGroups_dimensions_camera;
}

export interface GetDashboardData_currentUser_site_incidentAnalytics_incidentAggregateGroups_metrics {
  __typename: "IncidentAggregateMetrics";
  /**
   * Count of incidents in this group.
   */
  count: number;
}

export interface GetDashboardData_currentUser_site_incidentAnalytics_incidentAggregateGroups {
  __typename: "IncidentAggregateGroup";
  id: string;
  dimensions: GetDashboardData_currentUser_site_incidentAnalytics_incidentAggregateGroups_dimensions;
  metrics: GetDashboardData_currentUser_site_incidentAnalytics_incidentAggregateGroups_metrics;
}

export interface GetDashboardData_currentUser_site_incidentAnalytics {
  __typename: "SiteIncidentAnalytics";
  id: string;
  incidentAggregateGroups: GetDashboardData_currentUser_site_incidentAnalytics_incidentAggregateGroups[] | null;
}

export interface GetDashboardData_currentUser_site_productionLines_camera {
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

export interface GetDashboardData_currentUser_site_productionLines_status1hGroups_dimensions {
  __typename: "ProductionLineStatusDimension";
  /**
   * Production line group datetime truncated to the appropriate date part based on the group_by property (e.g. hourly groups are truncated to the hour, daily groups truncated to the day, etc.). All values are localized to the event zone's timezone.
   */
  datetime: any;
}

export interface GetDashboardData_currentUser_site_productionLines_status1hGroups_metrics {
  __typename: "ProductionLineStatusMetrics";
  /**
   * Duration of production line uptime (in seconds)
   */
  uptimeDurationSeconds: number;
  /**
   * Duration of production line downtime (in seconds)
   */
  downtimeDurationSeconds: number;
  /**
   * Duration of time where production line status is unknown (in seconds)
   */
  unknownDurationSeconds: number;
}

export interface GetDashboardData_currentUser_site_productionLines_status1hGroups {
  __typename: "ProductionLineStatusGroup";
  dimensions: GetDashboardData_currentUser_site_productionLines_status1hGroups_dimensions;
  metrics: GetDashboardData_currentUser_site_productionLines_status1hGroups_metrics;
}

export interface GetDashboardData_currentUser_site_productionLines {
  __typename: "ProductionLine";
  id: string;
  uuid: string;
  name: string;
  camera: GetDashboardData_currentUser_site_productionLines_camera;
  status1hGroups: GetDashboardData_currentUser_site_productionLines_status1hGroups[];
}

export interface GetDashboardData_currentUser_site {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  cameras: GetDashboardData_currentUser_site_cameras | null;
  overallScore: GetDashboardData_currentUser_site_overallScore | null;
  eventScores: (GetDashboardData_currentUser_site_eventScores | null)[] | null;
  latestActivityTimestamp: any | null;
  incidentCategories: (GetDashboardData_currentUser_site_incidentCategories | null)[] | null;
  assigneeStats: (GetDashboardData_currentUser_site_assigneeStats | null)[] | null;
  incidentAnalytics: GetDashboardData_currentUser_site_incidentAnalytics;
  productionLines: GetDashboardData_currentUser_site_productionLines[] | null;
}

export interface GetDashboardData_currentUser_tasksAssignedByStats {
  __typename: "TaskStatsType";
  totalCount: number | null;
  openCount: number | null;
  resolvedCount: number | null;
}

export interface GetDashboardData_currentUser_tasksAssignedToStats {
  __typename: "TaskStatsType";
  totalCount: number | null;
  openCount: number | null;
  resolvedCount: number | null;
}

export interface GetDashboardData_currentUser {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  stats: GetDashboardData_currentUser_stats | null;
  site: GetDashboardData_currentUser_site | null;
  tasksAssignedByStats: GetDashboardData_currentUser_tasksAssignedByStats | null;
  tasksAssignedToStats: GetDashboardData_currentUser_tasksAssignedToStats | null;
}

export interface GetDashboardData {
  currentUser: GetDashboardData_currentUser | null;
}

export interface GetDashboardDataVariables {
  startDate: any;
  endDate: any;
  startTimestamp: any;
  endTimestamp: any;
  groupBy: TimeBucketWidth;
}
