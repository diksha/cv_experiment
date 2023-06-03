import {
  HelpOutlineOutlined,
  Logout,
  PeopleOutlineOutlined,
  PersonOutline,
  ThumbUpOutlined,
} from "@mui/icons-material";
import { Paper, Typography, useTheme } from "@mui/material";
import { PAGE_REVIEW_QUEUE, useCurrentUser } from "features/auth";
import { ElementType, useState } from "react";
import { Helmet } from "react-helmet-async";
import { Link } from "react-router-dom";
import { useRouting } from "shared/hooks";
import { BackgroundSpinner, TopNavBack } from "ui";

export function AccountMenuPage() {
  const { currentUser, logout } = useCurrentUser();
  const theme = useTheme();
  const [loggingOut, setLoggingOut] = useState(false);

  const handleLogout = () => {
    setLoggingOut(true);
    try {
      logout({ returnTo: window.location.origin });
    } catch (error) {
      setLoggingOut(false);
      alert("Sorry, something went wrong while logging out.");
    }
  };

  if (loggingOut) {
    return <BackgroundSpinner />;
  }

  return (
    <>
      <Helmet>
        <title>Account - Voxel</title>
      </Helmet>
      <TopNavBack mobTitle="Settings" icon="logo" />
      <CustomRow title="My Profile" icon={PersonOutline} to="/account/profile" />
      <CustomRow title="Teammates" icon={PeopleOutlineOutlined} to="/account/teammates" />
      <CustomRow title="Support" icon={HelpOutlineOutlined} to="/support" />
      {currentUser?.hasGlobalPermission(PAGE_REVIEW_QUEUE) && (
        <CustomRow title="Review" icon={ThumbUpOutlined} to="/review" />
      )}
      <Paper
        sx={{
          padding: "24px",
          display: "flex",
          alignItems: "center",
          borderBottom: `1px solid ${theme.palette.grey[300]}`,
          cursor: "pointer",
          borderRadius: 0,
        }}
        onClick={handleLogout}
      >
        <Logout sx={{ height: 24, width: 24, marginRight: "24px" }} />
        <Typography variant="h4">Logout</Typography>
      </Paper>
    </>
  );
}

interface CustomRowProps {
  title: string;
  icon: ElementType;
  to: string;
}
function CustomRow(props: CustomRowProps) {
  const theme = useTheme();
  const { newLocationState } = useRouting();
  return (
    <Paper
      sx={{
        padding: "24px",
        display: "flex",
        alignItems: "center",
        borderBottom: `1px solid ${theme.palette.grey[300]}`,
        borderRadius: 0,
      }}
      component={Link}
      to={props.to}
      state={newLocationState}
    >
      <props.icon sx={{ height: 24, width: 24, marginRight: "24px" }} />
      <Typography variant="h4">{props.title}</Typography>
    </Paper>
  );
}
