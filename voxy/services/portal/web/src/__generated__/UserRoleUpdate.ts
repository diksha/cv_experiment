/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: UserRoleUpdate
// ====================================================

export interface UserRoleUpdate_userRoleUpdate {
  __typename: "UserRoleUpdate";
  status: boolean | null;
}

export interface UserRoleUpdate {
  userRoleUpdate: UserRoleUpdate_userRoleUpdate | null;
}

export interface UserRoleUpdateVariables {
  userId: string;
  roleId: string;
}
