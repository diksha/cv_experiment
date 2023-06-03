/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

import { FilterInputType, ApiIncidentPriorityChoices } from "./globalTypes";

// ====================================================
// GraphQL query operation: GetCurrentZoneIncidentFeed
// ====================================================

export interface GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_pageInfo {
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

export interface GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets_incidentCounts_incidentType {
  __typename: "IncidentTypeType";
  key: string;
  name: string;
}

export interface GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets_incidentCounts {
  __typename: "TimeBucketIncidentTypeCount";
  count: number | null;
  incidentType: GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets_incidentCounts_incidentType | null;
}

export interface GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets_latestIncidents_assignees {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  initials: string | null;
  fullName: string | null;
}

export interface GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets_latestIncidents_incidentType {
  __typename: "CameraIncidentType";
  id: string;
  key: string;
  name: string;
}

export interface GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets_latestIncidents_camera {
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

export interface GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets_latestIncidents {
  __typename: "IncidentType";
  /**
   * The ID of the object
   */
  id: string;
  uuid: string;
  title: string;
  timestamp: any;
  priority: ApiIncidentPriorityChoices;
  status: string | null;
  bookmarked: boolean;
  highlighted: boolean;
  alerted: boolean;
  assignees: (GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets_latestIncidents_assignees | null)[];
  incidentType: GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets_latestIncidents_incidentType | null;
  camera: GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets_latestIncidents_camera | null;
}

export interface GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets {
  __typename: "IncidentFeedItemTimeBucket";
  key: string;
  title: string;
  /**
   * Time bucket exact start timestamp
   */
  startTimestamp: any;
  /**
   * Time bucket exact end timestamp
   */
  endTimestamp: any;
  /**
   * Total incident count within this time bucket
   */
  incidentCount: number;
  /**
   * Total incident counts by type within time bucket
   */
  incidentCounts: (GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets_incidentCounts | null)[] | null;
  /**
   * Latest N incidents within this time bucket
   */
  latestIncidents: (GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets_latestIncidents | null)[] | null;
}

export interface GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem {
  __typename: "DailyIncidentsFeedItem";
  key: string;
  date: any;
  /**
   * Time buckets covering all 24 hours of a single day, localized to the organization/zone timezone
   */
  timeBuckets: (GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets | null)[];
}

export interface GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_EmptyRangeFeedItem {
  __typename: "EmptyRangeFeedItem";
  key: string;
  startDate: any;
  endDate: any;
}

export type GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node = GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem | GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_EmptyRangeFeedItem;

export interface GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges {
  __typename: "IncidentFeedItemEdge";
  cursor: string;
  node: GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node | null;
}

export interface GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed {
  __typename: "IncidentFeedConnection";
  pageInfo: GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_pageInfo;
  edges: (GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges | null)[] | null;
}

export interface GetCurrentZoneIncidentFeed_currentUser_zone {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  timezone: string;
  incidentFeed: GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed | null;
}

export interface GetCurrentZoneIncidentFeed_currentUser {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  zone: GetCurrentZoneIncidentFeed_currentUser_zone | null;
}

export interface GetCurrentZoneIncidentFeed {
  currentUser: GetCurrentZoneIncidentFeed_currentUser | null;
}

export interface GetCurrentZoneIncidentFeedVariables {
  startDate?: any | null;
  endDate?: any | null;
  filters?: (FilterInputType | null)[] | null;
  timeBucketSizeHours?: number | null;
  after?: string | null;
}
