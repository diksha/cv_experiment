import React, { useState, ElementType } from "react";
import { PAGE_REVIEW_QUEUE, useCurrentUser } from "features/auth";
import { Link } from "react-router-dom";
import { useTheme, Skeleton, Box, Avatar, Menu, MenuItem, ListItemIcon, Divider, IconButton } from "@mui/material";
import {
  Logout,
  HelpOutlineOutlined,
  PersonOutline,
  PeopleOutlineOutlined,
  ThumbUpOutlined,
} from "@mui/icons-material";
import { BackgroundSpinner } from "ui";
import { REVIEW_PATH } from "features/mission-control/review";
import { useRouting } from "shared/hooks";

const AVATAR_SIZE_PX = 32;

// Mostly taken from MUI examples: https://mui.com/material-ui/react-menu/#account-menu
export function AccountMenu() {
  const { currentUser, isLoading, logout } = useCurrentUser();
  const [loggingOut, setLoggingOut] = useState(false);
  const theme = useTheme();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);
  const hasReviewPermission = currentUser?.hasGlobalPermission(PAGE_REVIEW_QUEUE);

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

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

  if (isLoading) {
    return <AccountMenuSkeleton />;
  }

  return (
    <>
      <Box sx={{ display: "flex", alignItems: "center", textAlign: "center" }}>
        <IconButton
          onClick={handleClick}
          size="small"
          aria-controls={open ? "account-menu" : undefined}
          aria-haspopup="true"
          aria-expanded={open ? "true" : undefined}
        >
          <Avatar
            src={currentUser?.picture}
            alt={currentUser?.fullName}
            sx={{ height: AVATAR_SIZE_PX, width: AVATAR_SIZE_PX }}
          />
        </IconButton>
      </Box>
      <Menu
        anchorEl={anchorEl}
        id="account-menu"
        open={open}
        onClose={handleClose}
        onClick={handleClose}
        PaperProps={{
          elevation: 0,
          sx: {
            overflow: "visible",
            filter: "drop-shadow(0px 2px 8px rgba(0,0,0,0.32))",
            mt: 1.5,
            "& .MuiAvatar-root": {
              width: AVATAR_SIZE_PX,
              height: AVATAR_SIZE_PX,
              ml: -0.5,
              mr: 1,
            },
            // This is a trick to get a small triangle on the top of the menu
            "&:before": {
              content: '""',
              display: "block",
              position: "absolute",
              top: 0,
              right: 14,
              width: 10,
              height: 10,
              bgcolor: "background.paper",
              transform: "translateY(-50%) rotate(45deg)",
              zIndex: 0,
            },
          },
        }}
        transformOrigin={{ horizontal: "right", vertical: "top" }}
        anchorOrigin={{ horizontal: "right", vertical: "bottom" }}
      >
        <MenuLink to="/account/profile" icon={PersonOutline} label="My Profile" />
        <MenuLink to="/account/teammates" icon={PeopleOutlineOutlined} label="Teammates" />
        <MenuLink to="/support" icon={HelpOutlineOutlined} label="Support" />
        {hasReviewPermission && <MenuLink to={REVIEW_PATH} icon={ThumbUpOutlined} label="Review" />}
        <Divider sx={{ borderColor: theme.palette.grey[300] }} />
        <MenuItem onClick={handleLogout}>
          <ListItemIcon>
            <Logout fontSize="small" />
          </ListItemIcon>
          Logout
        </MenuItem>
      </Menu>
    </>
  );
}

interface MenuLinkProps {
  label: string;
  icon: ElementType;
  to: string;
}
function MenuLink({ label, icon, to }: MenuLinkProps) {
  const { newLocationState } = useRouting();
  const Icon = icon;
  return (
    <MenuItem component={Link} to={to} state={newLocationState}>
      <ListItemIcon>
        <Icon fontSize="small" />
      </ListItemIcon>
      <Box sx={{ paddingRight: 2 }}>{label}</Box>
    </MenuItem>
  );
}

function AccountMenuSkeleton() {
  return <Skeleton variant="circular" sx={{ height: AVATAR_SIZE_PX, width: AVATAR_SIZE_PX }} />;
}
