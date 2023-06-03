/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

import { ApiIncidentPriorityChoices } from "./globalTypes";

// ====================================================
// GraphQL query operation: GetIncidentDetails
// ====================================================

export interface GetIncidentDetails_incidentDetails_incidentType {
  __typename: "CameraIncidentType";
  id: string;
  key: string;
  name: string;
  backgroundColor: string;
}

export interface GetIncidentDetails_incidentDetails_camera {
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

export interface GetIncidentDetails_incidentDetails_zone {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  name: string;
}

export interface GetIncidentDetails_incidentDetails_tags {
  __typename: "TagType";
  label: string | null;
  value: string;
}

export interface GetIncidentDetails_incidentDetails_assignees {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  fullName: string | null;
}

export interface GetIncidentDetails_incidentDetails_organization {
  __typename: "OrganizationType";
  /**
   * The ID of the object
   */
  id: string;
  name: string;
}

export interface GetIncidentDetails_incidentDetails {
  __typename: "IncidentType";
  /**
   * The ID of the object
   */
  id: string;
  uuid: string;
  pk: number;
  title: string;
  timestamp: any;
  endTimestamp: any | null;
  incidentType: GetIncidentDetails_incidentDetails_incidentType | null;
  priority: ApiIncidentPriorityChoices;
  status: string | null;
  alerted: boolean;
  bookmarked: boolean;
  highlighted: boolean;
  videoUrl: string;
  annotationsUrl: string;
  actorIds: string[];
  camera: GetIncidentDetails_incidentDetails_camera | null;
  zone: GetIncidentDetails_incidentDetails_zone | null;
  tags: (GetIncidentDetails_incidentDetails_tags | null)[];
  assignees: (GetIncidentDetails_incidentDetails_assignees | null)[];
  organization: GetIncidentDetails_incidentDetails_organization | null;
  duration: number | null;
}

export interface GetIncidentDetails {
  incidentDetails: GetIncidentDetails_incidentDetails | null;
}

export interface GetIncidentDetailsVariables {
  incidentUuid: string;
}
