/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetFilteredDashboardData
// ====================================================

export interface GetFilteredDashboardData_currentUser_site_filteredIncidentPriorityStats {
  __typename: "ZoneIncidentStats";
  highPriorityCount: number | null;
  mediumPriorityCount: number | null;
  lowPriorityCount: number | null;
}

export interface GetFilteredDashboardData_currentUser_site_filteredIncidentTypeStats_incidentType {
  __typename: "IncidentTypeType";
  key: string;
  name: string;
  category: string | null;
  backgroundColor: string | null;
}

export interface GetFilteredDashboardData_currentUser_site_filteredIncidentTypeStats {
  __typename: "ZoneIncidentTypeStats";
  incidentType: GetFilteredDashboardData_currentUser_site_filteredIncidentTypeStats_incidentType | null;
  totalCount: number | null;
  /**
   * Highest totalCount value of all incident type stats returned in a query. Used by clients to compare totalCount with maxTotalCount for things like bar charts with relative widths per incident type.
   */
  maxTotalCount: number | null;
}

export interface GetFilteredDashboardData_currentUser_site_filteredCameraStats_camera {
  __typename: "CameraType";
  /**
   * The ID of the object
   */
  id: string;
  uuid: string;
  /**
   * User friendly name displayed throughout apps.
   */
  name: string;
}

export interface GetFilteredDashboardData_currentUser_site_filteredCameraStats_categoryStats {
  __typename: "IncidentCategoryStats";
  categoryKey: string | null;
  totalCount: number | null;
}

export interface GetFilteredDashboardData_currentUser_site_filteredCameraStats {
  __typename: "CameraStats";
  camera: GetFilteredDashboardData_currentUser_site_filteredCameraStats_camera | null;
  categoryStats: (GetFilteredDashboardData_currentUser_site_filteredCameraStats_categoryStats | null)[] | null;
}

export interface GetFilteredDashboardData_currentUser_site_filteredAssigneeStats_assignee {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  fullName: string | null;
}

export interface GetFilteredDashboardData_currentUser_site_filteredAssigneeStats {
  __typename: "AssigneeStats";
  assignee: GetFilteredDashboardData_currentUser_site_filteredAssigneeStats_assignee | null;
  openCount: number | null;
  resolvedCount: number | null;
  resolvedTimeAvgMinutes: number | null;
}

export interface GetFilteredDashboardData_currentUser_site_productionLines_camera {
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

export interface GetFilteredDashboardData_currentUser_site_productionLines_status1hGroups_dimensions {
  __typename: "ProductionLineStatusDimension";
  /**
   * Production line group datetime truncated to the appropriate date part based on the group_by property (e.g. hourly groups are truncated to the hour, daily groups truncated to the day, etc.). All values are localized to the event zone's timezone.
   */
  datetime: any;
}

export interface GetFilteredDashboardData_currentUser_site_productionLines_status1hGroups_metrics {
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

export interface GetFilteredDashboardData_currentUser_site_productionLines_status1hGroups {
  __typename: "ProductionLineStatusGroup";
  dimensions: GetFilteredDashboardData_currentUser_site_productionLines_status1hGroups_dimensions;
  metrics: GetFilteredDashboardData_currentUser_site_productionLines_status1hGroups_metrics;
}

export interface GetFilteredDashboardData_currentUser_site_productionLines {
  __typename: "ProductionLine";
  id: string;
  uuid: string;
  name: string;
  camera: GetFilteredDashboardData_currentUser_site_productionLines_camera;
  status1hGroups: GetFilteredDashboardData_currentUser_site_productionLines_status1hGroups[];
}

export interface GetFilteredDashboardData_currentUser_site {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  filteredIncidentPriorityStats: GetFilteredDashboardData_currentUser_site_filteredIncidentPriorityStats | null;
  filteredIncidentTypeStats: (GetFilteredDashboardData_currentUser_site_filteredIncidentTypeStats | null)[] | null;
  filteredCameraStats: (GetFilteredDashboardData_currentUser_site_filteredCameraStats | null)[] | null;
  filteredAssigneeStats: (GetFilteredDashboardData_currentUser_site_filteredAssigneeStats | null)[] | null;
  productionLines: GetFilteredDashboardData_currentUser_site_productionLines[] | null;
}

export interface GetFilteredDashboardData_currentUser {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  site: GetFilteredDashboardData_currentUser_site | null;
}

export interface GetFilteredDashboardData {
  currentUser: GetFilteredDashboardData_currentUser | null;
}

export interface GetFilteredDashboardDataVariables {
  startTimestamp: any;
  endTimestamp: any;
}
