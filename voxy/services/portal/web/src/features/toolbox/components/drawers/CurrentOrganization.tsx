import React, { useCallback, useMemo, useState } from "react";
import { Spinner } from "ui";
import { filterNullValues } from "shared/utilities/types";
import { useCurrentUser } from "features/auth";
import { Buildings, CheckCircle } from "phosphor-react";
import { GET_ALL_ORGANIZATIONS_AND_SITES, Drawer } from "features/toolbox";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery } from "@apollo/client";
import {
  GetAllOrganizationsAndSites,
  GetAllOrganizationsAndSites_organizations_edges_node,
  GetAllOrganizationsAndSites_organizations_edges_node_sites,
} from "__generated__/GetAllOrganizationsAndSites";
import { getNodes } from "graphql/utils";
import classNames from "classnames";
import { CurrentUserSiteUpdateVariables, CurrentUserSiteUpdate } from "__generated__/CurrentUserSiteUpdate";
import { CURRENT_USER_SITE_UPDATE } from "features/organizations";

interface OrganizationNode extends GetAllOrganizationsAndSites_organizations_edges_node {}
interface Site extends GetAllOrganizationsAndSites_organizations_edges_node_sites {}

function Content() {
  const navigate = useNavigate();
  const { currentUser } = useCurrentUser();
  const [updating, setUpdating] = useState(false);
  const { loading, error, data } = useQuery<GetAllOrganizationsAndSites>(GET_ALL_ORGANIZATIONS_AND_SITES);
  const [currentUserSiteUpdate] = useMutation<CurrentUserSiteUpdate, CurrentUserSiteUpdateVariables>(
    CURRENT_USER_SITE_UPDATE,
    {
      onCompleted: () => {
        // Refresh page
        navigate(0);
      },
      onError: () => {
        setUpdating(false);
        alert("Failed to update current site.");
      },
    }
  );

  const handleSiteChanged = useCallback(
    async (siteId: string) => {
      if (siteId) {
        setUpdating(true);
        await currentUserSiteUpdate({
          variables: {
            siteId,
          },
        });
      } else {
        alert("Sorry, something went wrong while switching sites");
      }
    },
    [currentUserSiteUpdate]
  );

  const orgs = useMemo(() => {
    return getNodes<OrganizationNode>(data?.organizations)
      .slice()
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [data]);

  const customerOrgs = useMemo(() => {
    return orgs.filter((org) => !org.isSandbox);
  }, [orgs]);

  const sandboxOrgs = useMemo(() => {
    return orgs.filter((org) => org.isSandbox);
  }, [orgs]);

  return (
    <div>
      {loading || updating ? (
        <Spinner white className="flex justify-center w-full p-8" />
      ) : (
        <div className="flex flex-col gap-4">
          <div className="text-lg font-bold">Sandbox Orgs ({sandboxOrgs.length})</div>
          {sandboxOrgs.map((org) => (
            <OrganizationCard
              key={org.id}
              organization={org}
              currentSiteId={currentUser?.site?.id}
              onSiteChanged={handleSiteChanged}
            />
          ))}
          <div className="text-lg font-bold">Customer Orgs ({customerOrgs.length})</div>
          {customerOrgs.map((org) => (
            <OrganizationCard
              key={org.id}
              organization={org}
              currentSiteId={currentUser?.site?.id}
              onSiteChanged={handleSiteChanged}
            />
          ))}
        </div>
      )}
      {error ? JSON.stringify(error) : null}
    </div>
  );
}
interface OrganizationCardProps {
  organization: OrganizationNode;
  currentSiteId?: string;
  onSiteChanged: (siteId: string) => void;
}

function OrganizationCard(props: OrganizationCardProps) {
  const { organization, currentSiteId, onSiteChanged } = props;
  const handleSiteChanged = (siteId: string) => {
    onSiteChanged(siteId);
  };
  const sites = useMemo(() => {
    const sites = filterNullValues<Site>(organization?.sites?.slice() || []);
    // Sort by active first, then alphabetical
    return sites.sort((a, b) => {
      const activeValue = (b.isActive ? 1 : 0) - (a.isActive ? 1 : 0);
      const nameValue = a?.name.localeCompare(b?.name);
      return activeValue || nameValue;
    });
  }, [organization]);

  return (
    <div className="flex flex-col gap-4 bg-brand-gray-400 p-3 rounded-md">
      <div className="font-bold">{organization.name}</div>
      {sites.map((site) => {
        const selected = site.id === currentSiteId;
        const buttonClasses = classNames(
          "flex gap-2 bg-brand-gray-300 p-2 rounded-md items-center border border-brand-gray-300",
          "hover:opacity-100 hover:border-brand-gray-100",
          {
            "opacity-50": !site.isActive,
            "border border-brand-green-500": selected,
          }
        );

        return (
          <button key={site?.id} className={buttonClasses} onClick={() => handleSiteChanged(site.id)}>
            <div className="flex-1 text-left">
              {site.name} {site.isActive ? "" : "(inactive)"}
            </div>
            {selected ? <CheckCircle className="h-6 w-6 text-brand-green-500" /> : null}
          </button>
        );
      })}
    </div>
  );
}

export function CurrentOrganization() {
  return (
    <Drawer name="Current organization" icon={<Buildings className="h-6 w-6 text-brand-orange-300" />}>
      <Content />
    </Drawer>
  );
}
