import { useCurrentUser, INTERNAL_SITE_ACCESS } from "features/auth";
import { useState, useMemo } from "react";
import { Helmet } from "react-helmet-async";
import { SuccessToast, ErrorToast, PageWrapper, Avatar, LoadingOverlay, TopNavBack } from "ui";
import { FormItem, USER_NAME_UPDATE, USER_MFA_UPDATE, TabbedContainer, AccountTab } from "features/account";
import { EditOutlined } from "@mui/icons-material";
import { LoadingButton } from "@mui/lab";
import {
  Box,
  Card,
  TextField,
  Typography,
  IconButton,
  Button,
  Dialog,
  Switch,
  FormControlLabel,
  useTheme,
  useMediaQuery,
} from "@mui/material";
import { useMutation } from "@apollo/client";
import { toast } from "react-hot-toast";

// TODO: remove Hack to prevent tailwind.base.css from overriding these styles
const TEXT_FIELD_INPUT_PROPS_TAILWIND_MUI_INTEROP_HACK = {
  type: "text",
  style: { padding: "15.5px 14px", fontSize: "100%", backgroundColor: "#fafafa" },
};
const MAX_NAME_LENGTH = 100;

export function ProfilePage() {
  const { currentUser, isLoading } = useCurrentUser();
  const theme = useTheme();
  const hasMFA = currentUser?.hasMFA || false;
  const [editNameModalOpen, setEditNameModalOpen] = useState(false);
  const [mfaModalOpen, setMFAModalOpen] = useState(false);
  const [wantsMFA, setWantsMFA] = useState(hasMFA);
  const showData = !isLoading && currentUser;
  const mdBreakpoint = useMediaQuery(theme.breakpoints.up("md"));

  const handleMFAModalSave = () => {
    setMFAModalOpen(false);
  };

  const handleMFAModalCancel = () => {
    setMFAModalOpen(false);

    // Reset the toggle after slight delay
    setTimeout(() => {
      setWantsMFA(hasMFA);
    }, 250);
  };

  const handleMFAChange = () => {
    setWantsMFA(!wantsMFA);
    setMFAModalOpen(true);
  };

  const roleList = useMemo(
    () => currentUser?.roles?.map((role) => <span key={role.name}>{role.name} </span>),
    [currentUser?.roles]
  );

  const handleEditNameClick = () => {
    setEditNameModalOpen(true);
  };

  const handleEditNameModalClose = () => {
    setEditNameModalOpen(false);
  };

  const content = (
    <div className="flex flex-col gap-4">
      <LoadingOverlay loading={isLoading} />
      {showData ? (
        <div>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {mdBreakpoint && (
              <Typography variant="h2" paddingY={2}>
                My Profile
              </Typography>
            )}
            <Avatar url={currentUser?.picture} name={currentUser?.fullName} />
            <FormItem title="Name">
              <Box sx={{ display: "flex", gap: 2 }}>
                <div>
                  {currentUser?.givenName} {currentUser?.familyName}
                </div>
                <IconButton
                  onClick={handleEditNameClick}
                  sx={{ margin: "-8px" }}
                  disabled={currentUser.connection === "google-oauth2"}
                >
                  <EditOutlined />
                </IconButton>
              </Box>
            </FormItem>
            <FormItem title="Email">{currentUser?.email}</FormItem>
            <FormItem title="Role">{roleList}</FormItem>
            {currentUser?.hasGlobalPermission(INTERNAL_SITE_ACCESS) ? (
              <FormItem title="Multi-factor Authentication (MFA)">
                <FormControlLabel control={<Switch checked={wantsMFA} onChange={handleMFAChange} />} label="Enable" />
              </FormItem>
            ) : null}
          </Box>
        </div>
      ) : null}
    </div>
  );

  return (
    <>
      <Helmet>
        <title>My Profile - Voxel</title>
      </Helmet>
      <TopNavBack mobTitle="My Profile" mobTo="/account" />
      <PageWrapper maxWidth="md" padding={mdBreakpoint ? 2 : 0} margin="0 auto">
        {mdBreakpoint ? (
          <TabbedContainer activeTab={AccountTab.Profile}>{content}</TabbedContainer>
        ) : (
          <Card sx={{ padding: 2, width: "100%", borderRadius: 0 }}>{content}</Card>
        )}
      </PageWrapper>
      <EditNameModal open={editNameModalOpen} onClose={handleEditNameModalClose} />
      <MFAConfirmationModal
        open={mfaModalOpen}
        onSave={handleMFAModalSave}
        onCancel={handleMFAModalCancel}
        hasMFA={hasMFA}
        wantsMFA={wantsMFA}
      />
    </>
  );
}

interface EditNameModalProps {
  open: boolean;
  onClose: () => void;
}

function EditNameModal({ open, onClose }: EditNameModalProps) {
  const { currentUser, isLoading: currentUserLoading, refresh } = useCurrentUser();
  const [firstName, setFirstName] = useState(currentUser?.givenName);
  const [lastName, setLastName] = useState(currentUser?.familyName);
  const [updateUserName, { loading: saving }] = useMutation(USER_NAME_UPDATE);
  const loading = currentUserLoading || saving;

  const [firstNameError, setFirstNameError] = useState("");
  const [lastNameError, setLastNameError] = useState("");

  const resetValidation = () => {
    setFirstNameError("");
    setLastNameError("");
  };

  const handleFirstNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFirstName(e.currentTarget.value);
  };

  const handleLastNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setLastName(e.currentTarget.value);
  };

  const validateForm = () => {
    resetValidation();
    let valid = true;

    if (!firstName) {
      setFirstNameError("First name is required");
      valid = false;
    }

    if (!lastName) {
      setLastNameError("Last name is required");
      valid = false;
    }

    if (firstName && firstName.length > MAX_NAME_LENGTH) {
      setFirstNameError(`Must be less than ${MAX_NAME_LENGTH} characters`);
      valid = false;
    }

    if (lastName && lastName.length > MAX_NAME_LENGTH) {
      setLastNameError(`Must be less than ${MAX_NAME_LENGTH} characters`);
      valid = false;
    }

    return valid;
  };

  const handleSave = async () => {
    if (!validateForm()) {
      return;
    }

    try {
      await updateUserName({
        variables: {
          userId: currentUser?.id,
          firstName: firstName,
          lastName: lastName,
        },
      });
      toast.custom((t) => <SuccessToast toast={t}>Name updated</SuccessToast>);
      refresh();
      onClose();
    } catch {
      toast.custom((t) => <ErrorToast toast={t}>Failed to update name</ErrorToast>);
    }
  };

  const handleClose = () => {
    // Reset form before closing
    setFirstName(currentUser?.givenName);
    setLastName(currentUser?.familyName);
    resetValidation();
    onClose();
  };

  return (
    <Dialog open={open} onClose={handleClose} PaperProps={{ sx: { width: "100%", maxWidth: "400px" } }}>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 3, padding: 4 }}>
        <Typography variant="h2">Edit Name</Typography>
        <TextField
          id="first-name-textfield"
          label="First Name"
          variant="outlined"
          type="text"
          error={!!firstNameError}
          helperText={firstNameError}
          // TODO: remove Hack to prevent tailwind.base.css from overriding these styles
          inputProps={TEXT_FIELD_INPUT_PROPS_TAILWIND_MUI_INTEROP_HACK}
          value={firstName}
          onChange={handleFirstNameChange}
        />
        <TextField
          id="last-name-textfield"
          label="Last Name"
          variant="outlined"
          type="text"
          error={!!lastNameError}
          helperText={lastNameError}
          // TODO: remove Hack to prevent tailwind.base.css from overriding these styles
          inputProps={TEXT_FIELD_INPUT_PROPS_TAILWIND_MUI_INTEROP_HACK}
          value={lastName}
          onChange={handleLastNameChange}
        />
        <Box sx={{ display: "flex", gap: 2, alignItems: "stretch" }}>
          <Button variant="outlined" onClick={handleClose} disabled={loading} sx={{ flex: "1" }}>
            Cancel
          </Button>
          <LoadingButton onClick={handleSave} variant="contained" loading={loading} sx={{ flex: "1" }}>
            Save
          </LoadingButton>
        </Box>
      </Box>
    </Dialog>
  );
}

interface MFAConfirmationModalProps {
  open: boolean;
  hasMFA: boolean;
  wantsMFA: boolean;
  onCancel: () => void;
  onSave: () => void;
}

function MFAConfirmationModal({ open, wantsMFA, onCancel, onSave }: MFAConfirmationModalProps) {
  const { currentUser, isLoading: currentUserLoading, refresh } = useCurrentUser();
  const [updateUserMFA, { loading: saving }] = useMutation(USER_MFA_UPDATE);
  const loading = currentUserLoading || saving;

  const handleSave = async () => {
    const successMessage = wantsMFA ? "MFA enabled" : "MFA disabled";
    const errorMessage = wantsMFA ? "Failed to enable MFA" : "Failed to disabled MFA";

    try {
      await updateUserMFA({
        variables: {
          userId: currentUser?.id,
          toggledMfaOn: wantsMFA,
        },
      });
      toast.custom((t) => <SuccessToast toast={t}>{successMessage}</SuccessToast>);
      onSave();
      await refresh();
    } catch (error) {
      toast.custom((t) => <ErrorToast toast={t}>{errorMessage}</ErrorToast>);
    }
  };

  return (
    <Dialog open={open} onClose={onCancel} PaperProps={{ sx: { width: "100%", maxWidth: "400px" } }}>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 3, padding: 4 }}>
        <Typography variant="h2">{wantsMFA ? "Enable MFA" : "Disable MFA"}</Typography>
        {wantsMFA ? (
          <div>
            Enabling MFA will log you out automatically and redirect you to the login page to register your
            authenticator.
          </div>
        ) : (
          <div>
            Disabling MFA will delete your saved authenticators. If you re-enable it later, you will be prompted to
            re-register an authenticator.
          </div>
        )}
        <Box sx={{ display: "flex", gap: 2, alignItems: "stretch" }}>
          <Button variant="outlined" onClick={onCancel} disabled={loading} sx={{ flex: "1" }}>
            Cancel
          </Button>
          <LoadingButton onClick={handleSave} variant="contained" loading={loading} sx={{ flex: "1" }}>
            Save
          </LoadingButton>
        </Box>
      </Box>
    </Dialog>
  );
}
