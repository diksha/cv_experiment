/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: UserResendInvitation
// ====================================================

export interface UserResendInvitation_userResendInvitation {
  __typename: "UserResendInvitation";
  status: boolean | null;
}

export interface UserResendInvitation {
  /**
   * Ability for a user to resend an invitation to a user to register.
   */
  userResendInvitation: UserResendInvitation_userResendInvitation | null;
}

export interface UserResendInvitationVariables {
  invitationToken: string;
}
