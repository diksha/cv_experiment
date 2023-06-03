/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetAllOrganizationsAndSites
// ====================================================

export interface GetAllOrganizationsAndSites_organizations_edges_node_sites {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  name: string;
  isActive: boolean;
}

export interface GetAllOrganizationsAndSites_organizations_edges_node {
  __typename: "OrganizationType";
  /**
   * The ID of the object
   */
  id: string;
  pk: number | null;
  name: string;
  isSandbox: boolean;
  sites: (GetAllOrganizationsAndSites_organizations_edges_node_sites | null)[] | null;
}

export interface GetAllOrganizationsAndSites_organizations_edges {
  __typename: "OrganizationTypeEdge";
  /**
   * A cursor for use in pagination
   */
  cursor: string;
  /**
   * The item at the end of the edge
   */
  node: GetAllOrganizationsAndSites_organizations_edges_node | null;
}

export interface GetAllOrganizationsAndSites_organizations {
  __typename: "OrganizationTypeConnection";
  /**
   * Contains the nodes in this connection.
   */
  edges: (GetAllOrganizationsAndSites_organizations_edges | null)[];
}

export interface GetAllOrganizationsAndSites {
  organizations: GetAllOrganizationsAndSites_organizations | null;
}
