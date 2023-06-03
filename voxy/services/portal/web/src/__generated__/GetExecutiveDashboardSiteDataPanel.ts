/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

import { TimeBucketWidth } from "./globalTypes";

// ====================================================
// GraphQL query operation: GetExecutiveDashboardSiteDataPanel
// ====================================================

export interface GetExecutiveDashboardSiteDataPanel_zone_incidentTypes {
  __typename: "IncidentTypeType";
  key: string;
  name: string;
  backgroundColor: string | null;
}

export interface GetExecutiveDashboardSiteDataPanel_zone_overallScore {
  __typename: "Score";
  label: string;
  value: number;
}

export interface GetExecutiveDashboardSiteDataPanel_zone_eventScores {
  __typename: "Score";
  label: string;
  value: number;
}

export interface GetExecutiveDashboardSiteDataPanel_zone_sessionCount_users_user_sites {
  __typename: "ZoneType";
  name: string;
}

export interface GetExecutiveDashboardSiteDataPanel_zone_sessionCount_users_user {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  email: string;
  fullName: string | null;
  picture: string | null;
  sites: (GetExecutiveDashboardSiteDataPanel_zone_sessionCount_users_user_sites | null)[] | null;
}

export interface GetExecutiveDashboardSiteDataPanel_zone_sessionCount_users {
  __typename: "SessionUserCount";
  user: GetExecutiveDashboardSiteDataPanel_zone_sessionCount_users_user | null;
  value: number;
}

export interface GetExecutiveDashboardSiteDataPanel_zone_sessionCount {
  __typename: "SessionCount";
  users: (GetExecutiveDashboardSiteDataPanel_zone_sessionCount_users | null)[] | null;
}

export interface GetExecutiveDashboardSiteDataPanel_zone_incidentAnalytics_incidentAggregateGroups_dimensions_incidentType {
  __typename: "OrganizationIncidentTypeType";
  id: string;
  key: string;
  name: string;
}

export interface GetExecutiveDashboardSiteDataPanel_zone_incidentAnalytics_incidentAggregateGroups_dimensions_camera {
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

export interface GetExecutiveDashboardSiteDataPanel_zone_incidentAnalytics_incidentAggregateGroups_dimensions {
  __typename: "IncidentAggregateDimensions";
  /**
   * Incident aggregate group datetime truncated to the appropriate date part based on the group_by property (e.g. hourly groups are truncated to the hour, daily groups truncated to the day, etc.). All values are localized to the parent zone's timezone.
   */
  datetime: any;
  incidentType: GetExecutiveDashboardSiteDataPanel_zone_incidentAnalytics_incidentAggregateGroups_dimensions_incidentType;
  camera: GetExecutiveDashboardSiteDataPanel_zone_incidentAnalytics_incidentAggregateGroups_dimensions_camera;
}

export interface GetExecutiveDashboardSiteDataPanel_zone_incidentAnalytics_incidentAggregateGroups_metrics {
  __typename: "IncidentAggregateMetrics";
  /**
   * Count of incidents in this group.
   */
  count: number;
}

export interface GetExecutiveDashboardSiteDataPanel_zone_incidentAnalytics_incidentAggregateGroups {
  __typename: "IncidentAggregateGroup";
  id: string;
  dimensions: GetExecutiveDashboardSiteDataPanel_zone_incidentAnalytics_incidentAggregateGroups_dimensions;
  metrics: GetExecutiveDashboardSiteDataPanel_zone_incidentAnalytics_incidentAggregateGroups_metrics;
}

export interface GetExecutiveDashboardSiteDataPanel_zone_incidentAnalytics {
  __typename: "SiteIncidentAnalytics";
  incidentAggregateGroups: GetExecutiveDashboardSiteDataPanel_zone_incidentAnalytics_incidentAggregateGroups[] | null;
}

export interface GetExecutiveDashboardSiteDataPanel_zone {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  key: string;
  name: string;
  timezone: string;
  isActive: boolean;
  incidentTypes: (GetExecutiveDashboardSiteDataPanel_zone_incidentTypes | null)[] | null;
  overallScore: GetExecutiveDashboardSiteDataPanel_zone_overallScore | null;
  eventScores: (GetExecutiveDashboardSiteDataPanel_zone_eventScores | null)[] | null;
  sessionCount: GetExecutiveDashboardSiteDataPanel_zone_sessionCount;
  incidentAnalytics: GetExecutiveDashboardSiteDataPanel_zone_incidentAnalytics;
}

export interface GetExecutiveDashboardSiteDataPanel {
  zone: GetExecutiveDashboardSiteDataPanel_zone | null;
}

export interface GetExecutiveDashboardSiteDataPanelVariables {
  zoneId?: string | null;
  startDate: any;
  endDate: any;
  groupBy: TimeBucketWidth;
}
