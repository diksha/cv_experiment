/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: IncidentUndoHighlight
// ====================================================

export interface IncidentUndoHighlight_incidentUndoHighlight_incident {
  __typename: "IncidentType";
  /**
   * The ID of the object
   */
  id: string;
  highlighted: boolean;
}

export interface IncidentUndoHighlight_incidentUndoHighlight {
  __typename: "IncidentUndoHighlight";
  incident: IncidentUndoHighlight_incidentUndoHighlight_incident | null;
}

export interface IncidentUndoHighlight {
  incidentUndoHighlight: IncidentUndoHighlight_incidentUndoHighlight | null;
}

export interface IncidentUndoHighlightVariables {
  incidentId: string;
}
