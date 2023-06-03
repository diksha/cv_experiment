/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: UserMfaUpdate
// ====================================================

export interface UserMfaUpdate_userMfaUpdate {
  __typename: "UserMFAUpdate";
  success: boolean | null;
}

export interface UserMfaUpdate {
  userMfaUpdate: UserMfaUpdate_userMfaUpdate | null;
}

export interface UserMfaUpdateVariables {
  userId: string;
  toggledMfaOn: boolean;
}
