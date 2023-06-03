/*
 * Copyright 2020-2021 Voxel Labs, Inc.
 * All rights reserved.
 *
 * This document may not be reproduced, republished, distributed, transmitted,
 * displayed, broadcast or otherwise exploited in any manner without the express
 * prior written permission of Voxel Labs, Inc. The receipt or possession of this
 * document does not convey any rights to reproduce, disclose, or distribute its
 * contents, or to manufacture, use, or sell anything that it may describe, in
 * whole or in part.
 */
import { JwtPayload } from "jwt-decode";
import { Permission } from "features/auth";
import {
  GetCurrentUserProfile_currentUser_roles,
  GetCurrentUserProfile_currentUser_site,
  GetCurrentUserProfile_currentUser_sites,
} from "__generated__/GetCurrentUserProfile";

export type Auth0User = {
  sub: string;
  email: string;
  email_verified: boolean;
  family_name: string;
  given_name: string;
  locale: string;
  name: string;
  nickname: string;
  picture: string;
  updated_at: string;
  has_mfa: boolean;
  connection: string;
};

export type User = {
  id: string;
  email: string;
  emailVerified: boolean;
  familyName: string;
  givenName: string;
  fullName: string;
  initials?: string;
  locale: string;
  picture: string;
  updatedAt: string;
  isActive: boolean;
  permissions?: Map<string, boolean>;
  roles?: GetCurrentUserProfile_currentUser_roles[];
};

interface ICurrentUser extends User {
  organization?: Organization;
  site?: GetCurrentUserProfile_currentUser_site;
  roles?: GetCurrentUserProfile_currentUser_roles[];
  sites?: (GetCurrentUserProfile_currentUser_sites | null)[];
}

export class CurrentUser implements ICurrentUser {
  id: string;
  email: string;
  emailVerified: boolean;
  familyName: string;
  givenName: string;
  fullName: string;
  locale: string;
  picture: string;
  updatedAt: string;
  isActive: boolean;
  isDemoEnvironment: boolean;
  permissions?: Map<string, boolean>;
  organization?: Organization;
  site?: GetCurrentUserProfile_currentUser_site;
  roles?: GetCurrentUserProfile_currentUser_roles[];
  sites?: (GetCurrentUserProfile_currentUser_sites | null)[];
  hasMFA: boolean;
  connection: string;

  constructor(
    auth0User: Auth0User,
    permissions: string[],
    roles: GetCurrentUserProfile_currentUser_roles[],
    organization?: Organization,
    site?: GetCurrentUserProfile_currentUser_site,
    sites?: (GetCurrentUserProfile_currentUser_sites | null)[]
  ) {
    this.id = auth0User.sub;
    this.email = auth0User.email;
    this.emailVerified = auth0User.email_verified;
    this.familyName = auth0User.family_name;
    this.givenName = auth0User.given_name;
    this.fullName = auth0User.name;
    this.locale = auth0User.locale;
    this.picture = auth0User.picture;
    this.updatedAt = auth0User.updated_at;
    this.hasMFA = auth0User.has_mfa || false;
    this.connection = auth0User.connection;
    // TODO: get this value from Auth0
    this.isActive = true;
    this.permissions = new Map(permissions.map((p) => [p, true]));
    this.roles = roles;
    this.organization = organization;
    this.site = site;
    this.sites = sites;
    this.isDemoEnvironment = (this.organization?.key || "") === "VOXEL_DEMO";
  }

  getBooleanPreference(key: string): boolean {
    const preferences = this?.site?.clientPreferences || [];
    return preferences.some((preference) => preference.key === key && Boolean(preference.value));
  }

  hasPermission(permission: string): boolean {
    return !!this.permissions?.get(permission);
  }

  hasZonePermission(permission: Permission): boolean {
    if (this.hasOrganizationPermission(permission)) {
      return true;
    }
    return !!permission.zonePermissionKey && this.hasPermission(permission.zonePermissionKey);
  }

  hasOrganizationPermission(permission: Permission): boolean {
    if (this.hasGlobalPermission(permission)) {
      return true;
    }
    return !!permission.organizationPermissionKey && this.hasPermission(permission.organizationPermissionKey);
  }

  hasGlobalPermission(permission: Permission): boolean {
    return !!permission.globalPermissionKey && this.hasPermission(permission.globalPermissionKey);
  }
}

export class AuthError {
  message: string;

  constructor(message: string) {
    this.message = message;
  }
}

export type AccessToken = JwtPayload & {
  permissions: string[];
};

// TODO:  remove the below types in favor of generated graphql query types
export interface Organization {
  id: string;
  name: string;
  key: string;
  roles?: Role[] | null;
}

export interface Role {
  id: string;
  name: string;
  key: string;
}

export interface Zone {
  id: string;
  name: string;
  key: string;
  timezone?: string;
}

export interface Profile {
  id: number;
  organization?: Organization;
  starListId?: number;
}
