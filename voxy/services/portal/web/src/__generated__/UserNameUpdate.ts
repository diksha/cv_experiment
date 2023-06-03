/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: UserNameUpdate
// ====================================================

export interface UserNameUpdate_userNameUpdate_user {
  __typename: "UserType";
  pk: number | null;
  firstName: string;
  lastName: string;
  email: string;
}

export interface UserNameUpdate_userNameUpdate {
  __typename: "UserNameUpdate";
  user: UserNameUpdate_userNameUpdate_user | null;
}

export interface UserNameUpdate {
  userNameUpdate: UserNameUpdate_userNameUpdate | null;
}

export interface UserNameUpdateVariables {
  userId: string;
  firstName?: string | null;
  lastName?: string | null;
}
