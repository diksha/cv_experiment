/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: UserRemove
// ====================================================

export interface UserRemove_userRemove {
  __typename: "UserRemove";
  status: boolean | null;
}

export interface UserRemove {
  userRemove: UserRemove_userRemove | null;
}

export interface UserRemoveVariables {
  userId: string;
}
