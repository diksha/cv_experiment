import React, { useState, useCallback } from "react";
import { ErrorToast, SuccessToast } from "ui";
import { Dialog, SelectChangeEvent, Button, FormControl, InputLabel, Select, MenuItem } from "@mui/material";
import { LoadingButton } from "@mui/lab";
import { GetCurrentUserTeammates_currentUser_teammates_edges_node } from "__generated__/GetCurrentUserTeammates";

import { useMutation } from "@apollo/client";

import { USER_ROLE_UPDATE } from "features/account";
import { GET_CURRENT_USER_TEAMMATES } from "features/organizations";

import { toast } from "react-hot-toast";
import { Role } from "features/auth";
import { UserRoleUpdate, UserRoleUpdateVariables } from "__generated__/UserRoleUpdate";

interface UserNode extends GetCurrentUserTeammates_currentUser_teammates_edges_node {}

export const EditRoleModal = (props: {
  isRoleModalOpen: boolean;
  roles: Role[] | undefined;
  onClose: () => void;
  fullName: string;
  selectedUser: UserNode | null;
}) => {
  const selectedUserRoleId = props.selectedUser?.roles?.[0]?.id || "";
  const [selectedRoleId, setSelectedRoleId] = useState<string>(selectedUserRoleId);

  const [userRoleUpdate, { loading }] = useMutation<UserRoleUpdate, UserRoleUpdateVariables>(USER_ROLE_UPDATE, {
    refetchQueries: [GET_CURRENT_USER_TEAMMATES],
  });

  const handleSubmit = useCallback(async () => {
    if (props.selectedUser?.id && selectedRoleId) {
      try {
        await userRoleUpdate({
          variables: {
            userId: props.selectedUser?.id,
            roleId: selectedRoleId,
          },
        });
        toast.custom((t) => (
          <SuccessToast toast={t}>
            Details for <span className="font-bold">{props.selectedUser?.fullName}</span> have been updated successfully
          </SuccessToast>
        ));
      } catch {
        toast.custom((t) => (
          <ErrorToast toast={t}>
            Failed to update details for <span className="font-bold">{props.selectedUser?.fullName}</span>
          </ErrorToast>
        ));
      } finally {
        props.onClose();
      }
    }
  }, [userRoleUpdate, selectedRoleId, props]);

  const handleRoleChange = (event: SelectChangeEvent<string>) => {
    if (event.target.value) {
      setSelectedRoleId(event.target.value);
    }
  };

  return (
    <Dialog open={props.isRoleModalOpen} onClose={props.onClose}>
      <div className="relative pt-6 px-6">
        <div className="pb-4 text-xl font-bold text-brand-gray-500 font-epilogue">Edit Role for {props.fullName}</div>
        <FormControl fullWidth>
          <InputLabel id="role-select-label">Role</InputLabel>
          <Select
            id="role-select"
            labelId="role-select-label"
            label="Role"
            value={selectedRoleId}
            onChange={handleRoleChange}
          >
            {props?.roles?.map((role) => (
              <MenuItem key={role.id} value={role.id}>
                {role.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <div className="pt-4 pb-4 grid grid-cols-2 gap-3">
          <Button variant="outlined" onClick={props.onClose} disabled={loading}>
            Cancel
          </Button>
          <LoadingButton variant="contained" onClick={handleSubmit} loading={loading}>
            Update
          </LoadingButton>
        </div>
      </div>
    </Dialog>
  );
};
