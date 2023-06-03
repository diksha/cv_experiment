import { LoadingButton } from "@mui/lab";
import { TextField, Dialog, Button } from "@mui/material";
import React, { useCallback, useEffect, useState } from "react";
import { SuccessToast, ErrorToast } from "ui";
import { toast } from "react-hot-toast";
import { useMutation } from "@apollo/client";
import { CAMERA_UPDATE, GET_SITE_CAMERAS } from "features/cameras";
import { CameraUpdate, CameraUpdateVariables } from "__generated__/CameraUpdate";

interface Props {
  cameraId: string;
  cameraName: string;
  open: boolean;
  bannedNames: string[];
  onClose: () => void;
}

function validation(value: string, bannedWords: string[]) {
  if (bannedWords.includes(value)) {
    return {
      error: true,
      message: "Camera name is already in use",
    };
  }
  if (value === "") {
    return {
      error: true,
      message: "Camera name can not be empty",
    };
  }
  return {
    error: false,
    message: "",
  };
}

export const CameraEditModal = (props: Props) => {
  const { open, bannedNames, onClose, cameraId, cameraName } = props;
  const [newCameraName, setNewCameraName] = useState<string>(cameraName);
  const [disabled, setDisabled] = useState<boolean>(true);
  const [error, setError] = useState("");
  const [cameraUpdate, { loading }] = useMutation<CameraUpdate, CameraUpdateVariables>(CAMERA_UPDATE, {
    refetchQueries: [GET_SITE_CAMERAS],
  });

  useEffect(() => {
    setNewCameraName(cameraName);
  }, [cameraName]);

  const handleClose = useCallback(async () => {
    await onClose();
    setError("");
    setNewCameraName("");
  }, [onClose, setError, setNewCameraName]);

  const handleSubmit = useCallback(async () => {
    try {
      await cameraUpdate({
        variables: {
          cameraId,
          cameraName: newCameraName,
        },
      });
      toast.custom((t) => <SuccessToast toast={t}>The camera name is updated</SuccessToast>);
    } catch {
      toast.custom((t) => <ErrorToast toast={t}>Failed to update camera name</ErrorToast>);
    } finally {
      handleClose();
    }
  }, [cameraUpdate, cameraId, newCameraName, handleClose]);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    event.preventDefault();
    const value = event.target.value;
    const { error, message } = validation(value, bannedNames);

    if (error) {
      setNewCameraName(value);
      setError(message);
      setDisabled(true);
    } else {
      setNewCameraName(value);
      setError("");
      setDisabled(false);
    }
  };

  return (
    <Dialog open={open} onClose={handleClose}>
      <div className="p-6 sm:w-auto md:w-auto lg:min-w-450">
        <div className="pb-6">
          <div className="font-epilogue text-xl font-bold">Edit Camera</div>
        </div>
        <div className="flex flex-col gap-4">
          <TextField
            id="camera-name-text"
            label="Camera Name"
            variant="outlined"
            type="text"
            error={!!error}
            // TODO: remove Hack to prevent tailwind.base.css from overriding these styles
            inputProps={{
              type: "text",
              style: { padding: "15.5px 14px", fontSize: "100%", backgroundColor: "#fafafa" },
            }}
            value={newCameraName}
            onChange={handleChange}
          />
          <div className="border-gray-300 grid grid-cols-2 gap-3 w-full">
            <Button variant="outlined" onClick={handleClose} disabled={loading}>
              Cancel
            </Button>
            <LoadingButton variant="contained" onClick={handleSubmit} disabled={disabled || loading} loading={loading}>
              Save
            </LoadingButton>
          </div>
        </div>
      </div>
    </Dialog>
  );
};
