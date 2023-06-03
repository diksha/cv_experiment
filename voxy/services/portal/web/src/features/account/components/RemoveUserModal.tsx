import { useMutation } from "@apollo/client";
import { USER_REMOVE } from "features/account";
import React, { useCallback } from "react";
import { ErrorToast, SuccessToast } from "ui";
import { Dialog, Button } from "@mui/material";
import { LoadingButton } from "@mui/lab";
import { GET_CURRENT_USER_TEAMMATES } from "features/organizations";
import { GetCurrentUserTeammates_currentUser_teammates_edges_node } from "__generated__/GetCurrentUserTeammates";
import { Trash } from "phosphor-react";

import { toast } from "react-hot-toast";
import { UserRemove, UserRemoveVariables } from "__generated__/UserRemove";
interface UserNode extends GetCurrentUserTeammates_currentUser_teammates_edges_node {}

export const RemoveUserModal = ({
  isRemoveModalOpened,
  onClose,
  selectedUser,
}: {
  isRemoveModalOpened: boolean;
  onClose: () => void;
  selectedUser: UserNode | null;
}) => {
  const [removeUser, { loading }] = useMutation<UserRemove, UserRemoveVariables>(USER_REMOVE, {
    refetchQueries: [GET_CURRENT_USER_TEAMMATES],
  });

  const handleClick = useCallback(async () => {
    if (selectedUser?.id) {
      try {
        await removeUser({
          variables: {
            userId: selectedUser?.id,
          },
        });
        toast.custom((t) => (
          <SuccessToast toast={t}>
            <span className="font-bold">{selectedUser?.fullName}</span> has been removed from Voxel
          </SuccessToast>
        ));
      } catch {
        toast.custom((t) => (
          <ErrorToast toast={t}>
            Failed to remove <span className="font-bold">{selectedUser?.fullName}</span>
          </ErrorToast>
        ));
      } finally {
        onClose();
      }
    }
  }, [selectedUser?.id, selectedUser?.fullName, onClose, removeUser]);

  return (
    <Dialog open={isRemoveModalOpened} onClose={onClose}>
      <div className="p-6 flex flex-col items-center">
        <div className="m-4 bg-brand-red-100 p-4 rounded-full">
          <Trash className="h-8 w-8 text-brand-red-600" />
        </div>
        <div className="text-xl font-bold text-brand-gray-500 font-epilogue">
          Remove {selectedUser?.fullName} from Voxel
        </div>
        <div className="text-sm text-brand-gray-200 p-2">
          Are you sure you want to remove {selectedUser?.fullName} from your Voxel account?
        </div>
      </div>
      <div className="border-gray-300 pt-4 pb-4 px-4 grid grid-cols-2 gap-3">
        <Button variant="outlined" onClick={onClose} disabled={loading}>
          Cancel
        </Button>
        <LoadingButton color="error" variant="contained" onClick={handleClick} loading={loading}>
          Remove
        </LoadingButton>
      </div>
    </Dialog>
  );
};
