import React, { useState, useMemo, useEffect, useCallback } from "react";
import { useMutation } from "@apollo/client";
import { filterNullValues } from "shared/utilities/types";
import { ErrorToast, SuccessToast } from "ui";
import { GetCurrentUserTeammates_currentUser_teammates_edges_node } from "__generated__/GetCurrentUserTeammates";
import { GetCurrentUserProfile_currentUser_sites } from "__generated__/GetCurrentUserProfile";
import { Dialog, Button } from "@mui/material";
import { LoadingButton } from "@mui/lab";
import { useCurrentUser } from "features/auth";
import { USER_ZONES_UPDATE } from "features/account";
import { GET_CURRENT_USER_TEAMMATES } from "features/organizations";
import { toast } from "react-hot-toast";
import classNames from "classnames";
import { UserZonesUpdate, UserZonesUpdateVariables } from "__generated__/UserZonesUpdate";

interface UserNode extends GetCurrentUserTeammates_currentUser_teammates_edges_node {}
interface SiteNode extends GetCurrentUserProfile_currentUser_sites {}
type SiteSelectionMap = Record<string, boolean>;

interface EditSiteModalProps {
  isSiteModalOpen: boolean;
  selectedUser: UserNode | null;
  onClose: () => void;
  className?: string;
}
export const EditSiteModal = (props: EditSiteModalProps) => {
  const [userZonesUpdate, { loading }] = useMutation<UserZonesUpdate, UserZonesUpdateVariables>(USER_ZONES_UPDATE, {
    refetchQueries: [GET_CURRENT_USER_TEAMMATES],
  });

  const [zoneSelections, setZoneSelections] = useState<SiteSelectionMap>({});
  const { currentUser } = useCurrentUser();
  const sites = useMemo(() => {
    return filterNullValues<SiteNode>(currentUser?.sites);
  }, [currentUser]);

  useEffect(() => {
    const selections = Object.fromEntries(sites.map((site) => [site.id, false]));
    props.selectedUser?.sites?.forEach((site) => {
      if (site && site.id in selections) {
        selections[site.id] = true;
      }
    });
    setZoneSelections(selections);
  }, [currentUser, sites, props.selectedUser, setZoneSelections]);

  const handleSubmit = useCallback(async () => {
    if (props.selectedUser?.id) {
      const submittedZones: string[] = [];
      for (const zone in zoneSelections) {
        if (zoneSelections[zone]) {
          submittedZones.push(zone);
        }
      }

      try {
        await userZonesUpdate({
          variables: {
            userId: props.selectedUser?.id,
            zones: submittedZones,
          },
        });

        toast.custom((t) => (
          <SuccessToast toast={t}>
            Site access has been updated for <span className="font-bold">{props.selectedUser?.fullName}</span>
          </SuccessToast>
        ));
      } catch {
        toast.custom((t) => (
          <ErrorToast toast={t}>
            Failed to update site access for <span className="font-bold">{props.selectedUser?.fullName}</span>
          </ErrorToast>
        ));
      } finally {
        props.onClose();
      }
    }
  }, [props, userZonesUpdate, zoneSelections]);

  const handleChange = useCallback((zoneId, checked) => {
    setZoneSelections((prevState) => {
      return {
        ...prevState,
        [zoneId]: checked,
      };
    });
  }, []);

  return (
    <Dialog open={props.isSiteModalOpen} onClose={props.onClose}>
      <div className="relative px-6 pt-6">
        <div className="flex items-center justify-between">
          <div className="text-xl font-bold text-brand-gray-500 font-epilogue">Edit Site Access</div>
        </div>
        <div className="py-2 text-sm text-brand-gray-300">
          Select the sites where <span className="font-bold">{props.selectedUser?.fullName}</span> should have access
        </div>
      </div>
      <EditSiteList sites={sites} zoneSelections={zoneSelections} onChange={handleChange} />
      <div className="p-6 grid grid-cols-2 gap-3">
        <Button variant="outlined" onClick={props.onClose} disabled={loading}>
          Cancel
        </Button>
        <LoadingButton variant="contained" onClick={handleSubmit} loading={loading}>
          Update
        </LoadingButton>
      </div>
    </Dialog>
  );
};

interface EditSiteListProps {
  sites: SiteNode[];
  zoneSelections: SiteSelectionMap;
  onChange: (zoneId: string, checked: boolean) => void;
  className?: string;
}
export const EditSiteList = ({ sites, zoneSelections, onChange, className }: EditSiteListProps) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(e.target.value, e.target.checked);
  };

  return (
    <div
      className={classNames(
        "px-6 overflow-y-scroll scrollbar-hidden max-h-96 divide-y divide-brand-gray-100",
        className
      )}
    >
      {sites?.map((site) => (
        <div key={site!.id}>
          <label
            className={classNames(
              "flex gap-2 items-center py-3 text-sm font-medium cursor-pointer",
              zoneSelections[site!.id] ? "text-brand-gray-900" : "text-brand-gray-300 hover:text-brand-gray-900"
            )}
          >
            {zoneSelections[site!.id]}
            <input
              type="checkbox"
              value={site?.id}
              onChange={handleChange}
              checked={zoneSelections[site!.id]}
              className="w-4 h-4 rounded-md focus:!ring-0 focus:ring-offset-0 checked:!bg-brand-purple-500"
            />
            <div>{site?.name}</div>
          </label>
        </div>
      ))}
    </div>
  );
};
