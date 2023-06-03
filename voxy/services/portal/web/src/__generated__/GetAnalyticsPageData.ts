/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

import { FilterInputType } from "./globalTypes";

// ====================================================
// GraphQL query operation: GetAnalyticsPageData
// ====================================================

export interface GetAnalyticsPageData_analytics_series_statusCounts {
  __typename: "StatusCountType";
  openCount: number;
  resolvedCount: number;
}

export interface GetAnalyticsPageData_analytics_series_priorityCounts {
  __typename: "PriorityCountType";
  lowPriorityCount: number;
  mediumPriorityCount: number;
  highPriorityCount: number;
}

export interface GetAnalyticsPageData_analytics_series {
  __typename: "SeriesType";
  key: any | null;
  incidentTypeCounts: any;
  statusCounts: GetAnalyticsPageData_analytics_series_statusCounts;
  priorityCounts: GetAnalyticsPageData_analytics_series_priorityCounts;
}

export interface GetAnalyticsPageData_analytics {
  __typename: "AnalyticsType";
  series: (GetAnalyticsPageData_analytics_series | null)[] | null;
}

export interface GetAnalyticsPageData_currentUser_site_incidentTypes {
  __typename: "IncidentTypeType";
  id: string;
  key: string;
  name: string;
  backgroundColor: string | null;
}

export interface GetAnalyticsPageData_currentUser_site {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  incidentTypes: (GetAnalyticsPageData_currentUser_site_incidentTypes | null)[] | null;
}

export interface GetAnalyticsPageData_currentUser {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  site: GetAnalyticsPageData_currentUser_site | null;
}

export interface GetAnalyticsPageData {
  analytics: GetAnalyticsPageData_analytics | null;
  currentUser: GetAnalyticsPageData_currentUser | null;
}

export interface GetAnalyticsPageDataVariables {
  startTimestamp: any;
  endTimestamp: any;
  groupBy: string;
  filters?: (FilterInputType | null)[] | null;
}
