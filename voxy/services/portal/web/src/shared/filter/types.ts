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

export const KEY_ASSIGNED_TO_ME = "assigned_to_me";
export const KEY_ASSIGNED_BY_ME = "assigned_by_me";

export const FILTER_KEY_PRIORITY = "PRIORITY";
export const FILTER_VALUE_HIGH_PRIORITY = "high";
export const FILTER_VALUE_MEDIUM_PRIORITY = "medium";
export const FILTER_VALUE_LOW_PRIORITY = "low";

export const FILTER_KEY_STATUS = "STATUS";
export const FILTER_VALUE_UNASSIGNED_STATUS = "UNASSIGNED_STATUS";
export const FILTER_VALUE_OPEN_AND_ASSIGNED_STATUS = "OPEN_AND_ASSIGNED_STATUS";
export const FILTER_VALUE_RESOLVED_STATUS = "RESOLVED_STATUS";

export const FILTER_KEY_ASSIGNMENT = "ASSIGNMENT";
export const FILTER_VALUE_ASSIGNED_TO_ME = "ASSIGNED_TO_ME";
export const FILTER_VALUE_ASSIGNED_BY_ME = "ASSIGNED_BY_ME";
export const FILTER_VALUE_BOOKMARKED = "BOOKMARKED";
export const FILTER_VALUE_HIGHLIGHTED = "HIGHLIGHTED";

export const FILTER_KEY_CAMERA = "CAMERA";

export const FILTER_KEY_INCIDENT_TYPE = "INCIDENT_TYPE";
export const FILTER_KEY_EXTRAS = "EXTRAS";

export const FILTER_STATUSES = {
  title: "Statuses",
  key: FILTER_KEY_STATUS,
  options: [
    {
      label: "Unassigned",
      value: FILTER_VALUE_UNASSIGNED_STATUS,
    },
    {
      label: "Open & Assigned",
      value: FILTER_VALUE_OPEN_AND_ASSIGNED_STATUS,
    },
    {
      label: "Resolved",
      value: FILTER_VALUE_RESOLVED_STATUS,
    },
  ],
};

export const FILTER_PRIORITIES = {
  title: "Priorities",
  key: FILTER_KEY_PRIORITY,
  options: [
    {
      label: "High",
      value: FILTER_VALUE_HIGH_PRIORITY,
    },
    {
      label: "Medium",
      value: FILTER_VALUE_MEDIUM_PRIORITY,
    },
    {
      label: "Low",
      value: FILTER_VALUE_LOW_PRIORITY,
    },
  ],
};

export const FILTER_ASSIGNEES = {
  title: "Assignees",
  key: FILTER_KEY_ASSIGNMENT,
  options: [
    {
      label: "Assigned To Me",
      value: FILTER_VALUE_ASSIGNED_TO_ME,
    },
    {
      label: "Assigned By Me",
      value: FILTER_VALUE_ASSIGNED_BY_ME,
    },
  ],
};

export const FILTER_EXTRAS = {
  title: "Extra Filters",
  key: FILTER_KEY_EXTRAS,
  options: [
    {
      label: "Bookmarked",
      value: FILTER_VALUE_BOOKMARKED,
    },
    {
      label: "Highlighted",
      value: FILTER_VALUE_HIGHLIGHTED,
    },
  ],
};

export interface FilterValue {
  value: string | string[] | boolean | null | undefined;
}

/**
 * FilterBag is a key/value store for filter values, which is the preferred
 * way to pass filters around between React components which care about
 * the filter values.
 */
export interface FilterBag {
  [key: string]: FilterValue | undefined;
}

export interface FilterFormSectionOption {
  label: string;
  value: string;
}

export interface FilterFormSectionOptionGroup {
  key: string;
  label?: string;
  options: FilterFormSectionOption[];
}

export interface FilterFormSectionDefinition {
  title: string;
  key: string;
  options?: FilterFormSectionOption[];
  groups?: FilterFormSectionOptionGroup[];
}

// This interface is required for sending "generic" filters to the backend
// because GraphQL currently doesn't not support union types as input types.
// So whenever that is available in the graphene library, we can replace this
// with something cleaner such as:
//
// {
//   field: string;
//   value: string | string[] | boolean;
// }
export interface SerializedFilter {
  key: string;
  valueJson: string;
}

export interface FilterSelectionOption {
  label: string;
  value: string;
}

export interface SerializeFiltersResponse {
  searchParams: URLSearchParams;
  serializedFilters: SerializedFilter[];
}

export interface ParseFiltersResponse {
  filterBag: FilterBag;
  serializedFilters: SerializedFilter[];
}

export interface FilterHookResponse {
  filterBag: FilterBag;
  setFilters: React.Dispatch<React.SetStateAction<FilterBag>>;
  serializedFilters: SerializedFilter[];
}

export interface FilterConfig {
  allFilters: boolean;
  incidentTypes: boolean;
  locations: boolean;
  assigned: boolean;
  bookmarked: boolean;
  highlighted: boolean;
  clearAll: boolean;
}
