/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetAllOrganizationsWithSites
// ====================================================

export interface GetAllOrganizationsWithSites_organizations_edges_node_sites {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  name: string;
}

export interface GetAllOrganizationsWithSites_organizations_edges_node {
  __typename: "OrganizationType";
  /**
   * The ID of the object
   */
  id: string;
  pk: number | null;
  name: string;
  sites: (GetAllOrganizationsWithSites_organizations_edges_node_sites | null)[] | null;
}

export interface GetAllOrganizationsWithSites_organizations_edges {
  __typename: "OrganizationTypeEdge";
  /**
   * A cursor for use in pagination
   */
  cursor: string;
  /**
   * The item at the end of the edge
   */
  node: GetAllOrganizationsWithSites_organizations_edges_node | null;
}

export interface GetAllOrganizationsWithSites_organizations {
  __typename: "OrganizationTypeConnection";
  /**
   * Contains the nodes in this connection.
   */
  edges: (GetAllOrganizationsWithSites_organizations_edges | null)[];
}

export interface GetAllOrganizationsWithSites {
  organizations: GetAllOrganizationsWithSites_organizations | null;
}
