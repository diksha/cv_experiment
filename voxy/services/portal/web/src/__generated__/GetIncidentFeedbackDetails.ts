/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetIncidentFeedbackDetails
// ====================================================

export interface GetIncidentFeedbackDetails_incidentDetails_feedback_user {
  __typename: "UserType";
  email: string;
}

export interface GetIncidentFeedbackDetails_incidentDetails_feedback {
  __typename: "IncidentFeedbackType";
  id: string;
  feedbackText: string | null;
  user: GetIncidentFeedbackDetails_incidentDetails_feedback_user;
}

export interface GetIncidentFeedbackDetails_incidentDetails_cameraConfig {
  __typename: "CameraConfigNewModelType";
  doors: any | null;
  drivingAreas: any | null;
  actionableRegions: any | null;
  intersections: any | null;
  endOfAisles: any | null;
  noPedestrianZones: any | null;
  noObstructionRegions: any | null;
}

export interface GetIncidentFeedbackDetails_incidentDetails {
  __typename: "IncidentType";
  /**
   * The ID of the object
   */
  id: string;
  pk: number;
  title: string;
  videoUrl: string;
  annotationsUrl: string;
  actorIds: string[];
  feedback: GetIncidentFeedbackDetails_incidentDetails_feedback[];
  cameraConfig: GetIncidentFeedbackDetails_incidentDetails_cameraConfig | null;
}

export interface GetIncidentFeedbackDetails {
  incidentDetails: GetIncidentFeedbackDetails_incidentDetails | null;
}

export interface GetIncidentFeedbackDetailsVariables {
  incidentId: string;
}
