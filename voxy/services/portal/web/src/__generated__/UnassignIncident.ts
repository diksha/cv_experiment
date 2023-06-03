/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: UnassignIncident
// ====================================================

export interface UnassignIncident_unassignIncident_assign {
  __typename: "AssignSchema";
  id: string;
}

export interface UnassignIncident_unassignIncident {
  __typename: "DeleteUserIncident";
  assign: UnassignIncident_unassignIncident_assign | null;
}

export interface UnassignIncident {
  unassignIncident: UnassignIncident_unassignIncident | null;
}

export interface UnassignIncidentVariables {
  incidentId: string;
  assigneeId: string;
}
