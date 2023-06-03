/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetAllOrganizations
// ====================================================

export interface GetAllOrganizations_organizations_edges_node {
  __typename: "OrganizationType";
  /**
   * The ID of the object
   */
  id: string;
  pk: number | null;
  name: string;
}

export interface GetAllOrganizations_organizations_edges {
  __typename: "OrganizationTypeEdge";
  /**
   * A cursor for use in pagination
   */
  cursor: string;
  /**
   * The item at the end of the edge
   */
  node: GetAllOrganizations_organizations_edges_node | null;
}

export interface GetAllOrganizations_organizations {
  __typename: "OrganizationTypeConnection";
  /**
   * Contains the nodes in this connection.
   */
  edges: (GetAllOrganizations_organizations_edges | null)[];
}

export interface GetAllOrganizations {
  organizations: GetAllOrganizations_organizations | null;
}
