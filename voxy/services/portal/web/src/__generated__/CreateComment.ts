/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

import { ApiCommentActivityTypeChoices } from "./globalTypes";

// ====================================================
// GraphQL mutation operation: CreateComment
// ====================================================

export interface CreateComment_createComment_comment_owner {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  firstName: string;
  lastName: string;
  picture: string | null;
}

export interface CreateComment_createComment_comment {
  __typename: "CommentType";
  /**
   * The ID of the object
   */
  id: string;
  text: string;
  createdAt: any;
  activityType: ApiCommentActivityTypeChoices | null;
  note: string | null;
  owner: CreateComment_createComment_comment_owner | null;
}

export interface CreateComment_createComment {
  __typename: "CreateComment";
  comment: CreateComment_createComment_comment | null;
}

export interface CreateComment {
  createComment: CreateComment_createComment | null;
}

export interface CreateCommentVariables {
  incidentId: string;
  text: string;
}
