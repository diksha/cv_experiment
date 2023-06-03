/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: UserZonesUpdate
// ====================================================

export interface UserZonesUpdate_userZonesUpdate {
  __typename: "UserZonesUpdate";
  status: boolean | null;
}

export interface UserZonesUpdate {
  userZonesUpdate: UserZonesUpdate_userZonesUpdate | null;
}

export interface UserZonesUpdateVariables {
  userId: string;
  zones: string[];
}
