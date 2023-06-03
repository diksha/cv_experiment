/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

import { TimeBucketWidth, FilterInputType } from "./globalTypes";

// ====================================================
// GraphQL query operation: GetExecutiveDashboardData
// ====================================================

export interface GetExecutiveDashboardData_currentUser_organization_incidentTypes {
  __typename: "OrganizationIncidentTypeType";
  key: string;
  name: string;
  backgroundColor: string;
}

export interface GetExecutiveDashboardData_currentUser_organization_overallScore {
  __typename: "Score";
  label: string;
  value: number;
}

export interface GetExecutiveDashboardData_currentUser_organization_eventScores {
  __typename: "Score";
  label: string;
  value: number;
}

export interface GetExecutiveDashboardData_currentUser_organization_sessionCount_users_user_sites {
  __typename: "ZoneType";
  name: string;
}

export interface GetExecutiveDashboardData_currentUser_organization_sessionCount_users_user {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  email: string;
  fullName: string | null;
  picture: string | null;
  sites: (GetExecutiveDashboardData_currentUser_organization_sessionCount_users_user_sites | null)[] | null;
}

export interface GetExecutiveDashboardData_currentUser_organization_sessionCount_users {
  __typename: "SessionUserCount";
  user: GetExecutiveDashboardData_currentUser_organization_sessionCount_users_user | null;
  value: number;
}

export interface GetExecutiveDashboardData_currentUser_organization_sessionCount_sites_site {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  name: string;
  isActive: boolean;
}

export interface GetExecutiveDashboardData_currentUser_organization_sessionCount_sites {
  __typename: "SessionSiteCount";
  site: GetExecutiveDashboardData_currentUser_organization_sessionCount_sites_site | null;
  value: number;
}

export interface GetExecutiveDashboardData_currentUser_organization_sessionCount {
  __typename: "SessionCount";
  users: (GetExecutiveDashboardData_currentUser_organization_sessionCount_users | null)[] | null;
  sites: (GetExecutiveDashboardData_currentUser_organization_sessionCount_sites | null)[] | null;
}

export interface GetExecutiveDashboardData_currentUser_organization_sites_overallScore {
  __typename: "Score";
  label: string;
  value: number;
}

export interface GetExecutiveDashboardData_currentUser_organization_sites_eventScores {
  __typename: "Score";
  label: string;
  value: number;
}

export interface GetExecutiveDashboardData_currentUser_organization_sites_incidentAnalytics_incidentAggregateGroups_dimensions_incidentType {
  __typename: "OrganizationIncidentTypeType";
  id: string;
  key: string;
  name: string;
}

export interface GetExecutiveDashboardData_currentUser_organization_sites_incidentAnalytics_incidentAggregateGroups_dimensions_camera {
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

export interface GetExecutiveDashboardData_currentUser_organization_sites_incidentAnalytics_incidentAggregateGroups_dimensions {
  __typename: "IncidentAggregateDimensions";
  /**
   * Incident aggregate group datetime truncated to the appropriate date part based on the group_by property (e.g. hourly groups are truncated to the hour, daily groups truncated to the day, etc.). All values are localized to the parent zone's timezone.
   */
  datetime: any;
  incidentType: GetExecutiveDashboardData_currentUser_organization_sites_incidentAnalytics_incidentAggregateGroups_dimensions_incidentType;
  camera: GetExecutiveDashboardData_currentUser_organization_sites_incidentAnalytics_incidentAggregateGroups_dimensions_camera;
}

export interface GetExecutiveDashboardData_currentUser_organization_sites_incidentAnalytics_incidentAggregateGroups_metrics {
  __typename: "IncidentAggregateMetrics";
  /**
   * Count of incidents in this group.
   */
  count: number;
}

export interface GetExecutiveDashboardData_currentUser_organization_sites_incidentAnalytics_incidentAggregateGroups {
  __typename: "IncidentAggregateGroup";
  id: string;
  dimensions: GetExecutiveDashboardData_currentUser_organization_sites_incidentAnalytics_incidentAggregateGroups_dimensions;
  metrics: GetExecutiveDashboardData_currentUser_organization_sites_incidentAnalytics_incidentAggregateGroups_metrics;
}

export interface GetExecutiveDashboardData_currentUser_organization_sites_incidentAnalytics {
  __typename: "SiteIncidentAnalytics";
  incidentAggregateGroups: GetExecutiveDashboardData_currentUser_organization_sites_incidentAnalytics_incidentAggregateGroups[] | null;
}

export interface GetExecutiveDashboardData_currentUser_organization_sites {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  key: string;
  name: string;
  timezone: string;
  isActive: boolean;
  overallScore: GetExecutiveDashboardData_currentUser_organization_sites_overallScore | null;
  eventScores: (GetExecutiveDashboardData_currentUser_organization_sites_eventScores | null)[] | null;
  incidentAnalytics: GetExecutiveDashboardData_currentUser_organization_sites_incidentAnalytics;
}

export interface GetExecutiveDashboardData_currentUser_organization {
  __typename: "OrganizationType";
  /**
   * The ID of the object
   */
  id: string;
  name: string;
  incidentTypes: GetExecutiveDashboardData_currentUser_organization_incidentTypes[];
  overallScore: GetExecutiveDashboardData_currentUser_organization_overallScore | null;
  eventScores: (GetExecutiveDashboardData_currentUser_organization_eventScores | null)[] | null;
  sessionCount: GetExecutiveDashboardData_currentUser_organization_sessionCount;
  sites: (GetExecutiveDashboardData_currentUser_organization_sites | null)[] | null;
}

export interface GetExecutiveDashboardData_currentUser {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  fullName: string | null;
  organization: GetExecutiveDashboardData_currentUser_organization | null;
}

export interface GetExecutiveDashboardData {
  currentUser: GetExecutiveDashboardData_currentUser | null;
}

export interface GetExecutiveDashboardDataVariables {
  startDate: any;
  endDate: any;
  groupBy: TimeBucketWidth;
  filters?: (FilterInputType | null)[] | null;
}
