/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetCurrentSiteData
// ====================================================

export interface GetCurrentSiteData_currentUser_site {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  key: string;
  name: string;
  isActive: boolean;
}

export interface GetCurrentSiteData_currentUser_sites {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  key: string;
  name: string;
  isActive: boolean;
}

export interface GetCurrentSiteData_currentUser_organization {
  __typename: "OrganizationType";
  /**
   * The ID of the object
   */
  id: string;
  name: string;
}

export interface GetCurrentSiteData_currentUser {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  site: GetCurrentSiteData_currentUser_site | null;
  sites: (GetCurrentSiteData_currentUser_sites | null)[] | null;
  organization: GetCurrentSiteData_currentUser_organization | null;
}

export interface GetCurrentSiteData {
  currentUser: GetCurrentSiteData_currentUser | null;
}
