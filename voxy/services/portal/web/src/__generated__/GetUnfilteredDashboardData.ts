/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetUnfilteredDashboardData
// ====================================================

export interface GetUnfilteredDashboardData_currentUser_stats {
  __typename: "UserStats";
  bookmarkTotalCount: number | null;
  bookmarkResolvedCount: number | null;
  bookmarkOpenCount: number | null;
}

export interface GetUnfilteredDashboardData_currentUser_site_allTimeIncidentStats {
  __typename: "ZoneIncidentStats";
  resolvedCount: number | null;
}

export interface GetUnfilteredDashboardData_currentUser_site_incidentCategories_incidentTypes {
  __typename: "IncidentTypeType";
  key: string;
  name: string;
}

export interface GetUnfilteredDashboardData_currentUser_site_incidentCategories {
  __typename: "IncidentCategory";
  key: string | null;
  name: string | null;
  incidentTypes: (GetUnfilteredDashboardData_currentUser_site_incidentCategories_incidentTypes | null)[] | null;
}

export interface GetUnfilteredDashboardData_currentUser_site {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  latestActivityTimestamp: any | null;
  allTimeIncidentStats: GetUnfilteredDashboardData_currentUser_site_allTimeIncidentStats | null;
  incidentCategories: (GetUnfilteredDashboardData_currentUser_site_incidentCategories | null)[] | null;
  highlightedEventsCount: number;
}

export interface GetUnfilteredDashboardData_currentUser_tasksAssignedByStats {
  __typename: "TaskStatsType";
  totalCount: number | null;
  openCount: number | null;
  resolvedCount: number | null;
}

export interface GetUnfilteredDashboardData_currentUser_tasksAssignedToStats {
  __typename: "TaskStatsType";
  totalCount: number | null;
  openCount: number | null;
  resolvedCount: number | null;
}

export interface GetUnfilteredDashboardData_currentUser {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  stats: GetUnfilteredDashboardData_currentUser_stats | null;
  site: GetUnfilteredDashboardData_currentUser_site | null;
  tasksAssignedByStats: GetUnfilteredDashboardData_currentUser_tasksAssignedByStats | null;
  tasksAssignedToStats: GetUnfilteredDashboardData_currentUser_tasksAssignedToStats | null;
}

export interface GetUnfilteredDashboardData {
  currentUser: GetUnfilteredDashboardData_currentUser | null;
}

export interface GetUnfilteredDashboardDataVariables {
  highlightedEventsStartTimestamp: any;
  highlightedEventsEndTimestamp: any;
}
