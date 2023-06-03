/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: IncidentCreateShareLink
// ====================================================

export interface IncidentCreateShareLink_incidentCreateShareLink {
  __typename: "IncidentCreateShareLink";
  shareLink: string | null;
}

export interface IncidentCreateShareLink {
  /**
   * Ability for a user to share an incident for viewing.
   */
  incidentCreateShareLink: IncidentCreateShareLink_incidentCreateShareLink | null;
}

export interface IncidentCreateShareLinkVariables {
  incidentId: string;
}
