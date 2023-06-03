/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: CurrentUserOrganizationUpdate
// ====================================================

export interface CurrentUserOrganizationUpdate_currentUserOrganizationUpdate {
  __typename: "CurrentUserOrganizationUpdate";
  status: boolean | null;
}

export interface CurrentUserOrganizationUpdate {
  currentUserOrganizationUpdate: CurrentUserOrganizationUpdate_currentUserOrganizationUpdate | null;
}

export interface CurrentUserOrganizationUpdateVariables {
  organizationId: string;
}
