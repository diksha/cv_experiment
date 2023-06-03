/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetProductionLines
// ====================================================

export interface GetProductionLines_currentUser_site_productionLines_camera {
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

export interface GetProductionLines_currentUser_site_productionLines_status1hGroups_dimensions {
  __typename: "ProductionLineStatusDimension";
  /**
   * Production line group datetime truncated to the appropriate date part based on the group_by property (e.g. hourly groups are truncated to the hour, daily groups truncated to the day, etc.). All values are localized to the event zone's timezone.
   */
  datetime: any;
}

export interface GetProductionLines_currentUser_site_productionLines_status1hGroups_metrics {
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

export interface GetProductionLines_currentUser_site_productionLines_status1hGroups {
  __typename: "ProductionLineStatusGroup";
  dimensions: GetProductionLines_currentUser_site_productionLines_status1hGroups_dimensions;
  metrics: GetProductionLines_currentUser_site_productionLines_status1hGroups_metrics;
}

export interface GetProductionLines_currentUser_site_productionLines {
  __typename: "ProductionLine";
  id: string;
  uuid: string;
  name: string;
  camera: GetProductionLines_currentUser_site_productionLines_camera;
  status1hGroups: GetProductionLines_currentUser_site_productionLines_status1hGroups[];
}

export interface GetProductionLines_currentUser_site {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  productionLines: GetProductionLines_currentUser_site_productionLines[] | null;
}

export interface GetProductionLines_currentUser {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  site: GetProductionLines_currentUser_site | null;
}

export interface GetProductionLines {
  currentUser: GetProductionLines_currentUser | null;
}

export interface GetProductionLinesVariables {
  startTimestamp: any;
  endTimestamp: any;
}
