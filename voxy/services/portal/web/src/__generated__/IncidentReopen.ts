/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: IncidentReopen
// ====================================================

export interface IncidentReopen_incidentReopen_incident {
  __typename: "IncidentType";
  /**
   * The ID of the object
   */
  id: string;
  status: string | null;
}

export interface IncidentReopen_incidentReopen {
  __typename: "IncidentReopen";
  incident: IncidentReopen_incidentReopen_incident | null;
}

export interface IncidentReopen {
  incidentReopen: IncidentReopen_incidentReopen | null;
}

export interface IncidentReopenVariables {
  incidentId: string;
}
