/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetExecutiveDashboardActiveEmployees
// ====================================================

export interface GetExecutiveDashboardActiveEmployees_currentUser_organization_sessionCount_users_user_sites {
  __typename: "ZoneType";
  name: string;
}

export interface GetExecutiveDashboardActiveEmployees_currentUser_organization_sessionCount_users_user {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  email: string;
  fullName: string | null;
  picture: string | null;
  sites: (GetExecutiveDashboardActiveEmployees_currentUser_organization_sessionCount_users_user_sites | null)[] | null;
}

export interface GetExecutiveDashboardActiveEmployees_currentUser_organization_sessionCount_users {
  __typename: "SessionUserCount";
  user: GetExecutiveDashboardActiveEmployees_currentUser_organization_sessionCount_users_user | null;
  value: number;
}

export interface GetExecutiveDashboardActiveEmployees_currentUser_organization_sessionCount {
  __typename: "SessionCount";
  users: (GetExecutiveDashboardActiveEmployees_currentUser_organization_sessionCount_users | null)[] | null;
}

export interface GetExecutiveDashboardActiveEmployees_currentUser_organization {
  __typename: "OrganizationType";
  /**
   * The ID of the object
   */
  id: string;
  name: string;
  sessionCount: GetExecutiveDashboardActiveEmployees_currentUser_organization_sessionCount;
}

export interface GetExecutiveDashboardActiveEmployees_currentUser {
  __typename: "UserType";
  /**
   * The ID of the object
   */
  id: string;
  organization: GetExecutiveDashboardActiveEmployees_currentUser_organization | null;
}

export interface GetExecutiveDashboardActiveEmployees {
  currentUser: GetExecutiveDashboardActiveEmployees_currentUser | null;
}

export interface GetExecutiveDashboardActiveEmployeesVariables {
  startDate: any;
  endDate: any;
}
