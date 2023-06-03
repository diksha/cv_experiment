import { Box, Button, Dialog, Typography, useTheme } from "@mui/material";
import { UserSession } from "features/executive-dashboard";
import { Avatar } from "ui";

interface UserInfoModalProps {
  open: boolean;
  userSession: UserSession;
  onClose: () => void;
}
export const UserInfoModal = ({ open, userSession, onClose }: UserInfoModalProps) => {
  const theme = useTheme();

  const sitesString = userSession.user?.sites
    ?.map((site) => {
      return site?.name;
    })
    .join(", ");

  return (
    <Dialog open={open} onClose={onClose}>
      {userSession ? (
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            maxWidth: "400px",
            padding: "24px 64px",
          }}
          data-ui-key="user-info-modal"
        >
          <Avatar url={userSession.user?.picture || ""} name={userSession.user?.fullName || ""} />
          <Typography variant="h4" marginY={1.5}>
            {userSession.user?.fullName}
          </Typography>
          <Typography marginBottom={3}>{userSession.user?.email}</Typography>
          <Typography marginBottom={1} sx={{ color: theme.palette.grey[500] }}>
            Sites
          </Typography>
          <Typography marginBottom={3} textAlign="center">
            {sitesString}
          </Typography>
          <Button onClick={onClose} sx={{ width: "100%" }} data-ui-key="user-info-modal-close-btn" variant="contained">
            Close
          </Button>
        </Box>
      ) : (
        <></>
      )}
    </Dialog>
  );
};
