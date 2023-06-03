/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

import { ApiCommentActivityTypeChoices } from "./globalTypes";

// ====================================================
// GraphQL query operation: GetComments
// ====================================================

export interface GetComments_comments_owner {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  fullName: string | null;
  picture: string | null;
}

export interface GetComments_comments {
  __typename: "CommentType";
  /**
   * The ID of the object
   */
  id: string;
  text: string;
  note: string | null;
  activityType: ApiCommentActivityTypeChoices | null;
  createdAt: any;
  owner: GetComments_comments_owner | null;
}

export interface GetComments {
  comments: (GetComments_comments | null)[] | null;
}

export interface GetCommentsVariables {
  incidentId: string;
}
