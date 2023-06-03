/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

import { FilterInputType } from "./globalTypes";

// ====================================================
// GraphQL query operation: GetCurrentZoneIncidentTypeStats
// ====================================================

export interface GetCurrentZoneIncidentTypeStats_currentUser_zone_incidentTypeStats_incidentType {
  __typename: "IncidentTypeType";
  key: string;
  name: string;
}

export interface GetCurrentZoneIncidentTypeStats_currentUser_zone_incidentTypeStats {
  __typename: "ZoneIncidentTypeStats";
  incidentType: GetCurrentZoneIncidentTypeStats_currentUser_zone_incidentTypeStats_incidentType | null;
  totalCount: number | null;
}

export interface GetCurrentZoneIncidentTypeStats_currentUser_zone {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  incidentTypeStats: (GetCurrentZoneIncidentTypeStats_currentUser_zone_incidentTypeStats | null)[] | null;
}

export interface GetCurrentZoneIncidentTypeStats_currentUser {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  zone: GetCurrentZoneIncidentTypeStats_currentUser_zone | null;
}

export interface GetCurrentZoneIncidentTypeStats {
  currentUser: GetCurrentZoneIncidentTypeStats_currentUser | null;
}

export interface GetCurrentZoneIncidentTypeStatsVariables {
  startTimestamp: any;
  endTimestamp: any;
  filters?: (FilterInputType | null)[] | null;
}
