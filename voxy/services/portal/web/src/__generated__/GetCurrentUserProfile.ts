/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetCurrentUserProfile
// ====================================================

export interface GetCurrentUserProfile_currentUser_roles {
  __typename: "Role";
  id: string;
  name: string;
  key: string;
}

export interface GetCurrentUserProfile_currentUser_organization_roles {
  __typename: "Role";
  id: string;
  name: string;
  key: string;
}

export interface GetCurrentUserProfile_currentUser_organization {
  __typename: "OrganizationType";
  /**
   * The ID of the object
   */
  id: string;
  name: string;
  key: string;
  roles: GetCurrentUserProfile_currentUser_organization_roles[] | null;
}

export interface GetCurrentUserProfile_currentUser_site_clientPreferences {
  __typename: "ClientPreference";
  key: string;
  value: string;
}

export interface GetCurrentUserProfile_currentUser_site {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  name: string;
  key: string;
  timezone: string;
  clientPreferences: GetCurrentUserProfile_currentUser_site_clientPreferences[];
}

export interface GetCurrentUserProfile_currentUser_sites {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  key: string;
  name: string;
}

export interface GetCurrentUserProfile_currentUser {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  roles: GetCurrentUserProfile_currentUser_roles[] | null;
  organization: GetCurrentUserProfile_currentUser_organization | null;
  site: GetCurrentUserProfile_currentUser_site | null;
  sites: (GetCurrentUserProfile_currentUser_sites | null)[] | null;
  permissions: string[] | null;
}

export interface GetCurrentUserProfile {
  currentUser: GetCurrentUserProfile_currentUser | null;
}
