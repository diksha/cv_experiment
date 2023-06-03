/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetIncidentTypes
// ====================================================

export interface GetIncidentTypes_incidentTypes {
  __typename: "IncidentTypeType";
  key: string;
  name: string;
  backgroundColor: string | null;
}

export interface GetIncidentTypes {
  incidentTypes: (GetIncidentTypes_incidentTypes | null)[];
}
