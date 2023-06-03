/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: IncidentHighlight
// ====================================================

export interface IncidentHighlight_incidentHighlight_incident {
  __typename: "IncidentType";
  /**
   * The ID of the object
   */
  id: string;
  highlighted: boolean;
}

export interface IncidentHighlight_incidentHighlight {
  __typename: "IncidentHighlight";
  incident: IncidentHighlight_incidentHighlight_incident | null;
}

export interface IncidentHighlight {
  incidentHighlight: IncidentHighlight_incidentHighlight | null;
}

export interface IncidentHighlightVariables {
  incidentId: string;
}
