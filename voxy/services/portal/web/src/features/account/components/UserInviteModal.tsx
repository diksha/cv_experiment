import { useMutation } from "@apollo/client";
import { USER_INVITE } from "features/account";
import React, { ReactNode, useMemo, useCallback, useState } from "react";
import { SuccessToast } from "ui";

import { Role } from "features/auth";
import { toast } from "react-hot-toast";
import { useCurrentUser } from "features/auth";
import {
  OutlinedInput,
  ListItemText,
  Checkbox,
  MenuItem,
  TextField,
  Dialog,
  SelectChangeEvent,
  FormControl,
  InputLabel,
  Select,
  IconButton,
  Button,
} from "@mui/material";
import { LoadingButton } from "@mui/lab";
import classNames from "classnames";
import { GET_CURRENT_USER_TEAMMATES } from "features/organizations";
import { UserPlus, Trash, PlusCircle } from "phosphor-react";

import { Zone } from "features/auth";
import { UserInvite, UserInviteVariables } from "__generated__/UserInvite";

interface Invitee {
  // Simple unique identifier for each instance
  readonly id: string;
  email: string;
  roleId: string;
  selectedSiteIds: string[];
  siteError: boolean;
  roleError: boolean;
  emailError: boolean;
}

function generateBlankInvitee(): Invitee {
  return {
    id: Math.random().toString(36).substring(2),
    email: "",
    roleId: "",
    selectedSiteIds: [],
    siteError: true,
    roleError: true,
    emailError: true,
  };
}

export const UserInviteModal = ({
  isUserInviteModalOpen,
  onClose,
}: {
  isUserInviteModalOpen: boolean;
  onClose: () => void;
}) => {
  const [inviteUser, { loading, error: apiError }] = useMutation<UserInvite, UserInviteVariables>(USER_INVITE, {
    refetchQueries: [GET_CURRENT_USER_TEAMMATES],
  });
  const [invitees, setInvitees] = useState<Invitee[]>([generateBlankInvitee()]);
  const [highlightErrors, setHighlightErrors] = useState(false);
  const [error, setError] = useState(false);
  const { currentUser } = useCurrentUser();
  const roles = currentUser?.organization?.roles!;

  const handleClose = useCallback(async () => {
    await onClose();
    setError(false);
    setInvitees([generateBlankInvitee()]);
  }, [onClose, setError, setInvitees]);

  const handleSubmit = useCallback(async () => {
    const errors = invitees.some((invitee) => invitee.emailError || invitee.roleError || invitee.siteError);

    setHighlightErrors(errors);
    setError(errors);

    if (errors) {
      return;
    }

    const serializedInvitees = invitees.map((invitee) => {
      return {
        email: invitee.email,
        roleId: invitee.roleId,
        zoneIds: invitee.selectedSiteIds,
      };
    });

    await inviteUser({
      variables: {
        invitees: serializedInvitees,
      },
    });

    handleClose();
    toast.custom((t) => (
      <SuccessToast toast={t}>
        {invitees.length > 1 ? "Invitations have" : "Invitation has"} been sent successfully
      </SuccessToast>
    ));
  }, [handleClose, invitees, inviteUser]);

  const handleInviteesChange = (invitees: Invitee[]) => {
    setInvitees(invitees);
  };

  const sites: Zone[] = useMemo(() => {
    const allSites = currentUser?.sites || [];
    return allSites.filter((s) => !!s) as Zone[];
  }, [currentUser]);

  return (
    <Dialog open={isUserInviteModalOpen} onClose={handleClose}>
      <div className="p-6">
        <div className="pb-6">
          <div className="font-epilogue text-xl font-bold">Invite Teammate</div>
          <div className="text-brand-gray-300">Invite your teammates to join your organization</div>
        </div>
        <div className="flex flex-col gap-4">
          <InviteeList
            invitees={invitees}
            roles={roles}
            sites={sites}
            onChange={handleInviteesChange}
            highlightErrors={highlightErrors}
          />
          {error ? <div className="text-brand-red-500">Please fill out all requested fields</div> : null}
          {apiError ? (
            <div className="text-brand-red-500">Something went wrong while sending invitations: {apiError.message}</div>
          ) : null}
          <div className="flex flex-row gap-4">
            <Button variant="outlined" onClick={handleClose} disabled={loading}>
              Cancel
            </Button>
            <LoadingButton
              loading={loading}
              loadingPosition="start"
              variant="contained"
              startIcon={<UserPlus />}
              onClick={handleSubmit}
            >
              Send Invite
            </LoadingButton>
          </div>
        </div>
      </div>
    </Dialog>
  );
};

function InviteeList(props: {
  invitees: Invitee[];
  roles: Role[];
  sites: Zone[];
  highlightErrors: boolean;
  onChange: (invitees: Invitee[]) => void;
}) {
  const handleAddInvitee = () => {
    props.onChange([...props.invitees, generateBlankInvitee()]);
  };

  const handleRemoveInvitee = (id: string) => {
    let invitees = [...props.invitees];
    const index = invitees.findIndex((invitee) => invitee.id === id);
    if (index === -1) {
      invitees = [generateBlankInvitee()];
    } else {
      invitees.splice(index, 1);
    }
    props.onChange(invitees);
  };

  const handleInviteeChanged = (updatedInvitee: Invitee) => {
    const invitees = [...props.invitees];
    const index = invitees.findIndex((invitee) => invitee.id === updatedInvitee.id);
    if (index > -1) {
      invitees[index] = updatedInvitee;
      props.onChange(invitees);
    }
  };

  return (
    <div>
      <div className="grid grid-cols-1 gap-8 md:gap-2 pb-3">
        {props.invitees.map((invitee) => (
          <InviteeRow
            key={invitee.id}
            invitee={invitee}
            roles={props.roles}
            sites={props.sites}
            allowRemove={props.invitees.length > 1}
            highlightErrors={props.highlightErrors}
            onChange={handleInviteeChanged}
            onRemove={handleRemoveInvitee}
          />
        ))}
      </div>
      <div className="hidden md:block">
        <AddInviteeButton onClick={handleAddInvitee} text="Add another" />
      </div>
    </div>
  );
}

function InviteeRow(props: {
  invitee: Invitee;
  allowRemove: boolean;
  roles: Role[];
  sites: Zone[];
  highlightErrors: boolean;
  onChange: (invitee: Invitee) => void;
  onRemove: (id: string) => void;
}) {
  const handleRemove = () => {
    props.onRemove(props.invitee.id);
  };

  const handleEmailChanged = (e: React.ChangeEvent<HTMLInputElement>) => {
    const changedValues = { email: e.target.value, emailError: !e.target.value.includes("@") };
    props.onChange({ ...props.invitee, ...changedValues });
  };

  const handleRoleChanged = (event: SelectChangeEvent<string>) => {
    const changedValues = { roleId: event.target.value || "", roleError: !event.target.value };
    props.onChange({ ...props.invitee, ...changedValues });
  };

  const handleSitesChanged = (selectedSiteIds: string | string[]) => {
    const selectedSiteIdsArray = typeof selectedSiteIds === "string" ? [selectedSiteIds] : selectedSiteIds;
    const invitee = {
      ...props.invitee,
      selectedSiteIds: selectedSiteIdsArray,
      siteError: selectedSiteIdsArray.length === 0,
    };
    props.onChange(invitee);
  };

  const selectedRole = props.roles.find((role) => role.id === props.invitee.roleId);

  return (
    <div
      className={classNames(
        "grid grid-cols-1 md:grid-cols-3 gap-2",
        "border-b-2 border-brand-gray-100 pb-8 last:border-b-0 last:pb-0 md:border-b-0 md:pb-0"
      )}
    >
      <TextField
        id={`invitee-email-${props.invitee.id}`}
        label="Email"
        variant="outlined"
        type="text"
        error={props.highlightErrors && props.invitee.emailError}
        // TODO: remove Hack to prevent tailwind.base.css from overriding these styles
        inputProps={{ type: "text", style: { padding: "15.5px 14px", fontSize: "100%", backgroundColor: "#fafafa" } }}
        value={props.invitee.email}
        onChange={handleEmailChanged}
      />
      <FormControl error={props.highlightErrors && props.invitee.roleError}>
        <InputLabel id="role-select-label">Role</InputLabel>
        <Select
          id="role-select"
          labelId="role-select-label"
          label="Role"
          value={selectedRole?.id || ""}
          onChange={handleRoleChanged}
        >
          {props.roles.map((role) => (
            <MenuItem key={role.id} value={role.id}>
              {role.name}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      <div className="flex flex-col md:flex-row gap-2 w-full items-center">
        <SiteDropdown
          sites={props.sites}
          selectedSiteIds={props.invitee.selectedSiteIds}
          onChange={handleSitesChanged}
          error={props.highlightErrors && props.invitee.siteError}
        />
        {props.allowRemove ? (
          <div>
            <IconButton aria-label="Remove invitee" onClick={handleRemove}>
              <Trash />
            </IconButton>
          </div>
        ) : null}
      </div>
    </div>
  );
}

const AddInviteeButton = (props: { onClick: () => void; text: string }) => (
  <Button onClick={props.onClick} startIcon={<PlusCircle />}>
    <span>{props.text}</span>
  </Button>
);

const SiteDropdown = ({
  sites,
  selectedSiteIds,
  onChange,
  error,
}: {
  sites: Zone[];
  selectedSiteIds: string[];
  onChange: (selectedSiteIds: string | string[]) => void;
  error: boolean;
}) => {
  const handleChange = (event: SelectChangeEvent<string[]>) => {
    onChange(event.target.value);
  };

  const renderValue = useCallback(
    (selected: string[]): ReactNode => {
      if (selected.length > 1) {
        return `${selected.length} sites`;
      } else if (selected.length === 1) {
        const selectedSite = sites.find((site) => site.id === selected[0]);
        if (selectedSite) {
          return selectedSite.name;
        }
      }
    },
    [sites]
  );

  return (
    <FormControl fullWidth error={!!error}>
      <InputLabel id="site-multi-select-label">Sites</InputLabel>
      <Select
        labelId="site-multi-select-label"
        id="site-multi-select"
        multiple
        value={selectedSiteIds}
        onChange={handleChange}
        input={<OutlinedInput label="Sites" />}
        renderValue={renderValue}
        error={!!error}
      >
        {sites.map((site) => (
          <MenuItem key={site.id} value={site.id}>
            <Checkbox checked={selectedSiteIds.includes(site.id)} />
            <ListItemText primary={site.name} />
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
};
