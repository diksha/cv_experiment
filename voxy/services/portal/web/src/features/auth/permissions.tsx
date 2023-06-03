export class Permission {
  capabilityKey: string;
  globalScope: boolean;
  organizationScope: boolean;
  zoneScope: boolean;
  globalPermissionKey: string | null;
  organizationPermissionKey: string | null;
  zonePermissionKey: string | null;

  constructor(
    capabilityKey: string,
    globalScope: boolean = true,
    organizationScope: boolean = true,
    zoneScope: boolean = true
  ) {
    this.capabilityKey = capabilityKey;
    this.globalScope = globalScope;
    this.organizationScope = organizationScope;
    this.zoneScope = zoneScope;
    this.globalPermissionKey = globalScope ? `global:${capabilityKey}` : null;
    this.organizationPermissionKey = organizationScope ? `organization:${capabilityKey}` : null;
    this.zonePermissionKey = globalScope ? `zone:${capabilityKey}` : null;
  }
}

// ****************************************************************************
// Pages
// ****************************************************************************
export const PAGE_DASHBOARD = new Permission("page:dashboard");
export const PAGE_EXECUTIVE_DASHBOARD = new Permission("page:executive_dashboard");
export const PAGE_ANALYTICS = new Permission("page:analytics");
export const PAGE_INCIDENTS = new Permission("page:incidents");
export const PAGE_INCIDENT_DETAILS = new Permission("page:incident_details");
export const PAGE_REVIEW_QUEUE = new Permission("page:review_queue");
export const PAGE_REVIEW_USERS = new Permission("page:review_users");
export const PAGE_REVIEW_HISTORY = new Permission("page:review_history");
export const PAGE_ACCOUNT = new Permission("page:account");
export const PAGE_ACCOUNT_CAMERAS = new Permission("page:account_cameras");

// ****************************************************************************
// Incidents
// ****************************************************************************
export const INCIDENTS_READ = new Permission("incidents:read");
export const INCIDENTS_ASSIGN = new Permission("incidents:assign");
export const INCIDENTS_UNASSIGN = new Permission("incidents:unassign");
export const INCIDENTS_RESOLVE = new Permission("incidents:resolve");
export const INCIDENTS_REOPEN = new Permission("incidents:reopen");
export const INCIDENTS_COMMENT = new Permission("incidents:comment");
export const INCIDENTS_BOOKMARK = new Permission("incidents:bookmark");
export const INCIDENTS_HIGHLIGHT = new Permission("incidents:highlight");
export const INCIDENTS_PUBLIC_LINK_CREATE = new Permission("incidents:public_link_create");
export const INCIDENTS_DOWNLOAD_VIDEO = new Permission("incidents:download_video");
export const EXPERIMENTAL_INCIDENTS_READ = new Permission("experimental_incidents:read");
export const INCIDENT_DETAILS_READ = new Permission("incident_details:read");

// ****************************************************************************
// Incident Feedback & Review Queue
// ****************************************************************************
export const INCIDENT_FEEDBACK_CREATE = new Permission("incident_feedback:create");
export const INCIDENT_FEEDBACK_READ = new Permission("incident_feedback:read");
export const REVIEW_QUEUE_READ = new Permission("review_queue:read", true, false, false);

// ****************************************************************************
// Users
// ****************************************************************************
export const USERS_INVITE = new Permission("users:invite");
export const USERS_REMOVE = new Permission("users:remove");
export const USERS_UPDATE_PROFILE = new Permission("users:update_profile");
export const USERS_UPDATE_ROLE = new Permission("users:update_role");
export const USERS_UPDATE_SITE = new Permission("users:update_site");
export const USERS_READ = new Permission("users:read");

// ****************************************************************************
// Zones
// ****************************************************************************
export const CAMERAS_RENAME = new Permission("cameras:rename");
export const CAMERAS_READ = new Permission("cameras:read");

// ****************************************************************************
// Downtime
// ****************************************************************************
export const DOWNTIME_READ = new Permission("downtime:read");

// ****************************************************************************
// Uncategorized
// ****************************************************************************
export const TOOLBOX = new Permission("toolbox", true, false, false);
export const SELF_SWITCH_ORGANIZATION = new Permission("self:switch_organization", true, true, false);
export const SELF_UPDATE_ROLE = new Permission("self:update_role");
export const INTERNAL_SITE_ACCESS = new Permission("self:internal_site_access", true, false, false);
export const SELF_UPDATE_PROFILE = new Permission("self:update_profile", true, false, false);
