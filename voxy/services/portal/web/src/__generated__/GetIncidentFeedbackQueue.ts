/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

import { ReviewQueueContext, ApiIncidentPriorityChoices } from "./globalTypes";

// ====================================================
// GraphQL query operation: GetIncidentFeedbackQueue
// ====================================================

export interface GetIncidentFeedbackQueue_incidentFeedbackQueue_organization {
  __typename: "OrganizationType";
  /**
   * The ID of the object
   */
  id: string;
  name: string;
}

export interface GetIncidentFeedbackQueue_incidentFeedbackQueue_zone {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  name: string;
}

export interface GetIncidentFeedbackQueue_incidentFeedbackQueue_cameraConfig {
  __typename: "CameraConfigNewModelType";
  doors: any | null;
  drivingAreas: any | null;
  actionableRegions: any | null;
  intersections: any | null;
  endOfAisles: any | null;
  noPedestrianZones: any | null;
  noObstructionRegions: any | null;
}

export interface GetIncidentFeedbackQueue_incidentFeedbackQueue_incidentType {
  __typename: "CameraIncidentType";
  key: string;
  description: string | null;
}

export interface GetIncidentFeedbackQueue_incidentFeedbackQueue {
  __typename: "IncidentType";
  /**
   * The ID of the object
   */
  id: string;
  uuid: string;
  pk: number;
  title: string;
  priority: ApiIncidentPriorityChoices;
  status: string | null;
  videoUrl: string;
  annotationsUrl: string;
  actorIds: string[];
  organization: GetIncidentFeedbackQueue_incidentFeedbackQueue_organization | null;
  zone: GetIncidentFeedbackQueue_incidentFeedbackQueue_zone | null;
  dockerImageTag: string | null;
  cameraUuid: string;
  cameraConfig: GetIncidentFeedbackQueue_incidentFeedbackQueue_cameraConfig | null;
  incidentType: GetIncidentFeedbackQueue_incidentFeedbackQueue_incidentType | null;
}

export interface GetIncidentFeedbackQueue {
  incidentFeedbackQueue: (GetIncidentFeedbackQueue_incidentFeedbackQueue | null)[];
}

export interface GetIncidentFeedbackQueueVariables {
  reviewQueueContext: ReviewQueueContext;
}
