/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: IncidentResolve
// ====================================================

export interface IncidentResolve_incidentResolve_incident {
  __typename: "IncidentType";
  /**
   * The ID of the object
   */
  id: string;
  status: string | null;
}

export interface IncidentResolve_incidentResolve {
  __typename: "IncidentResolve";
  incident: IncidentResolve_incidentResolve_incident | null;
}

export interface IncidentResolve {
  incidentResolve: IncidentResolve_incidentResolve | null;
}

export interface IncidentResolveVariables {
  incidentId: string;
}
