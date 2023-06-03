/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: CurrentUserSiteUpdate
// ====================================================

export interface CurrentUserSiteUpdate_currentUserSiteUpdate {
  __typename: "CurrentUserSiteUpdate";
  status: boolean | null;
  newSiteId: string | null;
}

export interface CurrentUserSiteUpdate {
  currentUserSiteUpdate: CurrentUserSiteUpdate_currentUserSiteUpdate | null;
}

export interface CurrentUserSiteUpdateVariables {
  siteId: string;
}
