/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: CurrentUserAddBookmark
// ====================================================

export interface CurrentUserAddBookmark_currentUserAddBookmark_incident {
  __typename: "IncidentType";
  /**
   * The ID of the object
   */
  id: string;
  bookmarked: boolean;
}

export interface CurrentUserAddBookmark_currentUserAddBookmark {
  __typename: "CurrentUserAddBookmark";
  incident: CurrentUserAddBookmark_currentUserAddBookmark_incident | null;
}

export interface CurrentUserAddBookmark {
  currentUserAddBookmark: CurrentUserAddBookmark_currentUserAddBookmark | null;
}

export interface CurrentUserAddBookmarkVariables {
  incidentId: string;
}
