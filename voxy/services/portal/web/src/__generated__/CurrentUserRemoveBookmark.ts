/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: CurrentUserRemoveBookmark
// ====================================================

export interface CurrentUserRemoveBookmark_currentUserRemoveBookmark_incident {
  __typename: "IncidentType";
  /**
   * The ID of the object
   */
  id: string;
  bookmarked: boolean;
}

export interface CurrentUserRemoveBookmark_currentUserRemoveBookmark {
  __typename: "CurrentUserRemoveBookmark";
  incident: CurrentUserRemoveBookmark_currentUserRemoveBookmark_incident | null;
}

export interface CurrentUserRemoveBookmark {
  currentUserRemoveBookmark: CurrentUserRemoveBookmark_currentUserRemoveBookmark | null;
}

export interface CurrentUserRemoveBookmarkVariables {
  incidentId: string;
}
