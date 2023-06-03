/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: UserUpdate
// ====================================================

export interface UserUpdate_userUpdate_user {
  __typename: "UserType";
  pk: number | null;
  firstName: string;
  lastName: string;
  email: string;
}

export interface UserUpdate_userUpdate {
  __typename: "UserUpdate";
  user: UserUpdate_userUpdate_user | null;
}

export interface UserUpdate {
  userUpdate: UserUpdate_userUpdate | null;
}

export interface UserUpdateVariables {
  userId: string;
  firstName?: string | null;
  lastName?: string | null;
  roles?: (string | null)[] | null;
  isActive?: boolean | null;
}
