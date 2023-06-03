/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetCurrentOrganizationIncidentTypes
// ====================================================

export interface GetCurrentOrganizationIncidentTypes_currentUser_organization_incidentTypes {
  __typename: "OrganizationIncidentTypeType";
  key: string;
  name: string;
  backgroundColor: string;
}

export interface GetCurrentOrganizationIncidentTypes_currentUser_organization {
  __typename: "OrganizationType";
  /**
   * The ID of the object
   */
  id: string;
  incidentTypes: GetCurrentOrganizationIncidentTypes_currentUser_organization_incidentTypes[];
}

export interface GetCurrentOrganizationIncidentTypes_currentUser {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  organization: GetCurrentOrganizationIncidentTypes_currentUser_organization | null;
}

export interface GetCurrentOrganizationIncidentTypes {
  currentUser: GetCurrentOrganizationIncidentTypes_currentUser | null;
}
