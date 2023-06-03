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
import React, { useMemo } from "react";
import { FilterOptions } from "shared/types";
import { Dropdown } from "ui";
import { getNodes } from "graphql/utils";
import { useQuery } from "@apollo/client";
import { GET_ALL_INCIDENT_TYPES } from "features/incidents/queries";
import { GET_ALL_ORGANIZATIONS_WITH_SITES } from "features/mission-control/review";
import {
  GetAllOrganizationsWithSites,
  GetAllOrganizationsWithSites_organizations_edges_node,
} from "__generated__/GetAllOrganizationsWithSites";
import { GetIncidentTypes, GetIncidentTypes_incidentTypes } from "__generated__/GetIncidentTypes";

interface OrganizationNode extends GetAllOrganizationsWithSites_organizations_edges_node {}

function mapOptions(incidentTypes: (GetIncidentTypes_incidentTypes | null)[] | undefined) {
  if (incidentTypes) {
    return [
      { value: "all", label: "All types", default: true },
      ...incidentTypes.map((item) => ({
        value: item?.key || "",
        label: item?.name || "",
      })),
    ];
  }
  return [];
}

export function IncidentTypeFilter({
  selectedValues,
  fetchOnChange,
  onChange,
}: {
  selectedValues?: string[];
  fetchOnChange?: boolean;
  onChange: (value: string, options: FilterOptions) => void;
}) {
  const { loading, data } = useQuery<GetIncidentTypes>(GET_ALL_INCIDENT_TYPES);
  const selected = (selectedValues || []).length > 0 ? selectedValues![0] : undefined;

  return (
    <Dropdown
      selectedValue={selected}
      loading={loading}
      options={mapOptions(data?.incidentTypes)}
      fetchOnChange={fetchOnChange}
      onChange={onChange}
      wrapperStyle="w-full md:w-auto"
      buttonStyle="w-full md:w-auto"
    />
  );
}

export interface OrganizationSiteFilterOption {
  value: string;
  label: string;
  organizationId?: string;
  siteId?: string;
  default?: boolean;
}

const OrganizationSiteFilterDefaultValue = {
  value: "all",
  label: "Organizations / Sites",
  organizationId: "all",
  siteId: "all",
  default: true,
};

export function OrganizationSiteFilter(props: {
  selectedValues?: OrganizationSiteFilterOption[];
  fetchOnChange?: boolean;
  onChange: (value: OrganizationSiteFilterOption, options: FilterOptions) => void;
}) {
  const { loading, data } = useQuery<GetAllOrganizationsWithSites>(GET_ALL_ORGANIZATIONS_WITH_SITES);
  const selected = (props.selectedValues || []).length > 0 ? props.selectedValues![0].value : undefined;

  const dropdownOptions = useMemo(() => {
    const options: OrganizationSiteFilterOption[] = [OrganizationSiteFilterDefaultValue];

    getNodes<OrganizationNode>(data?.organizations).forEach((organization) => {
      // Add options for organization + all sites
      options.push({
        value: `org:${organization.id};site:*`,
        label: `${organization.name} > All sites`,
        organizationId: organization.id,
      });

      (organization.sites || []).forEach((site) => {
        if (site?.id) {
          options.push({
            value: `org:${organization.id};site:${site.id}`,
            label: `${organization.name} > ${site.name}`,
            organizationId: organization.id,
            siteId: site.id,
          });
        }
      });
    });

    return options;
  }, [data]);

  const handleChange = (value: string, options: FilterOptions) => {
    const selected = dropdownOptions.find((option) => option.value === value);
    if (selected) {
      props.onChange(selected, options);
    }
  };

  return (
    <Dropdown
      selectedValue={selected}
      loading={loading}
      options={dropdownOptions}
      fetchOnChange={props.fetchOnChange}
      onChange={handleChange}
      wrapperStyle="w-full md:w-auto"
      buttonStyle="w-full md:w-auto"
    />
  );
}

OrganizationSiteFilter.DefaultValue = OrganizationSiteFilterDefaultValue;
