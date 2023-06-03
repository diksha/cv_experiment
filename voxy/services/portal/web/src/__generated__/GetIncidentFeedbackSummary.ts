/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetIncidentFeedbackSummary
// ====================================================

export interface GetIncidentFeedbackSummary_incidentFeedbackSummary_edges_node {
  __typename: "IncidentFeedbackSummaryType";
  /**
   * The ID of the object
   */
  id: string;
  uuid: any | null;
  valid: (string | null)[];
  invalid: (string | null)[];
  unsure: (string | null)[];
  lastFeedbackSubmissionTimestamp: any | null;
}

export interface GetIncidentFeedbackSummary_incidentFeedbackSummary_edges {
  __typename: "IncidentFeedbackSummaryTypeEdge";
  /**
   * A cursor for use in pagination
   */
  cursor: string;
  /**
   * The item at the end of the edge
   */
  node: GetIncidentFeedbackSummary_incidentFeedbackSummary_edges_node | null;
}

export interface GetIncidentFeedbackSummary_incidentFeedbackSummary_pageInfo {
  __typename: "PageInfo";
  /**
   * When paginating backwards, the cursor to continue.
   */
  startCursor: string | null;
  /**
   * When paginating forwards, the cursor to continue.
   */
  endCursor: string | null;
  /**
   * When paginating forwards, are there more items?
   */
  hasNextPage: boolean;
  /**
   * When paginating backwards, are there more items?
   */
  hasPreviousPage: boolean;
}

export interface GetIncidentFeedbackSummary_incidentFeedbackSummary {
  __typename: "IncidentFeedbackSummaryTypeConnection";
  /**
   * Contains the nodes in this connection.
   */
  edges: (GetIncidentFeedbackSummary_incidentFeedbackSummary_edges | null)[];
  /**
   * Pagination data for this connection.
   */
  pageInfo: GetIncidentFeedbackSummary_incidentFeedbackSummary_pageInfo;
}

export interface GetIncidentFeedbackSummary {
  incidentFeedbackSummary: GetIncidentFeedbackSummary_incidentFeedbackSummary | null;
}

export interface GetIncidentFeedbackSummaryVariables {
  internalFeedback?: string | null;
  externalFeedback?: string | null;
  hasComments?: boolean | null;
  incidentType?: string | null;
  organizationId?: string | null;
  siteId?: string | null;
  fromUtc?: any | null;
  toUtc?: any | null;
  first?: number | null;
  after?: string | null;
}
