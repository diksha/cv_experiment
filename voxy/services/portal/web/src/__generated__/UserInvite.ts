/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

import { InvitationInputSchema } from "./globalTypes";

// ====================================================
// GraphQL mutation operation: UserInvite
// ====================================================

export interface UserInvite_userInvite {
  __typename: "UserInvite";
  status: boolean | null;
}

export interface UserInvite {
  /**
   * Ability for a user to invite a user to register.
   */
  userInvite: UserInvite_userInvite | null;
}

export interface UserInviteVariables {
  invitees: (InvitationInputSchema | null)[];
}
