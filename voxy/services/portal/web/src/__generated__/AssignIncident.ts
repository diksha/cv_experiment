/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: AssignIncident
// ====================================================

export interface AssignIncident_assignIncident_users {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  firstName: string;
  lastName: string;
}

export interface AssignIncident_assignIncident {
  __typename: "CreateUserIncident";
  users: (AssignIncident_assignIncident_users | null)[] | null;
}

export interface AssignIncident {
  assignIncident: AssignIncident_assignIncident | null;
}

export interface AssignIncidentVariables {
  incidentId: string;
  assigneeIds?: (string | null)[] | null;
  note: string;
}
