/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: CreateIncidentFeedback
// ====================================================

export interface CreateIncidentFeedback_createIncidentFeedback_incidentFeedback {
  __typename: "IncidentFeedbackType";
  id: string;
  feedbackType: string;
  feedbackValue: string;
  feedbackText: string | null;
  updatedAt: any;
}

export interface CreateIncidentFeedback_createIncidentFeedback_userErrors {
  __typename: "UserError";
  message: string;
  code: string | null;
}

export interface CreateIncidentFeedback_createIncidentFeedback {
  __typename: "CreateIncidentFeedback";
  incidentFeedback: CreateIncidentFeedback_createIncidentFeedback_incidentFeedback | null;
  userErrors: (CreateIncidentFeedback_createIncidentFeedback_userErrors | null)[] | null;
}

export interface CreateIncidentFeedback {
  createIncidentFeedback: CreateIncidentFeedback_createIncidentFeedback | null;
}

export interface CreateIncidentFeedbackVariables {
  incidentId: string;
  feedbackType: string;
  feedbackValue: string;
  feedbackText: string;
  elapsedMillisecondsBetweenReviews?: number | null;
  incidentServedTimestampSeconds?: number | null;
}
