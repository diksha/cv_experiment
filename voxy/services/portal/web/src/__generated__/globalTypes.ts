/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

//==============================================================
// START Enums and Input Objects
//==============================================================

/**
 * An enumeration.
 */
export enum ApiCommentActivityTypeChoices {
  ASSIGN = "ASSIGN",
  COMMENT = "COMMENT",
  LOG = "LOG",
  REOPEN = "REOPEN",
  RESOLVE = "RESOLVE",
}

/**
 * An enumeration.
 */
export enum ApiIncidentPriorityChoices {
  HIGH = "HIGH",
  LOW = "LOW",
  MEDIUM = "MEDIUM",
}

/**
 * An enumeration.
 */
export enum ScenarioType {
  NEGATIVE = "NEGATIVE",
  POSITIVE = "POSITIVE",
}

/**
 * Time bucket width choices.
 */
export enum TimeBucketWidth {
  DAY = "DAY",
  HOUR = "HOUR",
  MONTH = "MONTH",
  QUARTER = "QUARTER",
  WEEK = "WEEK",
  YEAR = "YEAR",
}

export interface FilterInputType {
  key?: string | null;
  valueJson?: string | null;
}

export interface InvitationInputSchema {
  email: string;
  roleId: string;
  zoneIds: (string | null)[];
}

export interface ReviewQueueContext {
  reviewPanelId?: number | null;
  incidentExclusionList?: (string | null)[] | null;
}

//==============================================================
// END Enums and Input Objects
//==============================================================
