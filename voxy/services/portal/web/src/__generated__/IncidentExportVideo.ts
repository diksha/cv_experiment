/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: IncidentExportVideo
// ====================================================

export interface IncidentExportVideo_incidentExportVideo {
  __typename: "IncidentExportVideo";
  downloadUrl: string | null;
}

export interface IncidentExportVideo {
  incidentExportVideo: IncidentExportVideo_incidentExportVideo | null;
}

export interface IncidentExportVideoVariables {
  incidentId: string;
  labeled: boolean;
}
