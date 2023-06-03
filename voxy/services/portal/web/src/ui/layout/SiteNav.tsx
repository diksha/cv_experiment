import {
  PAGE_INCIDENTS,
  PAGE_ANALYTICS,
  PAGE_DASHBOARD,
  PAGE_EXECUTIVE_DASHBOARD,
  PAGE_ACCOUNT_CAMERAS,
} from "features/auth";
import { Toolbar, useTheme, useMediaQuery, Skeleton, Box, Drawer } from "@mui/material";
import { useCurrentUser } from "features/auth";
import {
  HomeOutlined,
  ManageSearchOutlined,
  MenuOutlined,
  TrendingUpOutlined,
  VideocamOutlined,
} from "@mui/icons-material";

import { NAV_ITEM_WIDTH, NAV_GAP, NAV_WIDTH, NAV_HEIGHT } from "./constants";
import { NavItem } from "./NavItem";
import { useRouting } from "shared/hooks";

interface SiteNavProps {
  mobileNavOnly?: boolean;
  hideRoutes?: boolean;
}
export function SiteNav({ mobileNavOnly, hideRoutes }: SiteNavProps) {
  const { currentUser, isLoading } = useCurrentUser();
  const theme = useTheme();
  const { fromExec } = useRouting();
  const mdBreakpoint = useMediaQuery(theme.breakpoints.up("md"));
  const ready = !isLoading && currentUser;
  const ResponsiveDrawer = mdBreakpoint ? DesktopDrawer : MobileDrawer;
  const hideExecRoutes = hideRoutes && fromExec;

  const navContent = () => {
    if (!ready) {
      return <LeftNavItemSkeletons />;
    }
    return (
      <>
        {currentUser.hasGlobalPermission(PAGE_EXECUTIVE_DASHBOARD) ? (
          <NavItem
            to={hideExecRoutes ? "/executive-dashboard" : "/dashboard"}
            text="Home"
            uiKey="menu-home"
            icon={HomeOutlined}
            extraMatches={["/bookmarks", "/tasks", "/dashboard"]}
          />
        ) : currentUser?.hasGlobalPermission(PAGE_DASHBOARD) ? (
          <NavItem
            to="/dashboard"
            text="Home"
            uiKey="menu-home"
            icon={HomeOutlined}
            extraMatches={["/bookmarks", "/tasks"]}
          />
        ) : (
          <></>
        )}
        {currentUser?.hasGlobalPermission(PAGE_INCIDENTS) && !hideExecRoutes && (
          <NavItem to="/incidents" text="Incidents" uiKey="menu-incidents" icon={ManageSearchOutlined} />
        )}
        {currentUser?.hasGlobalPermission(PAGE_ANALYTICS) && !hideExecRoutes && (
          <NavItem to="/analytics" text="Analytics" uiKey="menu-analytics" icon={TrendingUpOutlined} />
        )}
        {currentUser?.hasGlobalPermission(PAGE_ACCOUNT_CAMERAS) && !hideExecRoutes && (
          <NavItem to="/cameras" text="Cameras" uiKey="menu-cameras" icon={VideocamOutlined} />
        )}
        {currentUser && !mdBreakpoint && (
          <NavItem to="/account" text="Account" uiKey="menu-account" icon={MenuOutlined} extraMatches={["/support"]} />
        )}
      </>
    );
  };

  if (mdBreakpoint && mobileNavOnly) {
    return <></>;
  }
  return <ResponsiveDrawer>{navContent()}</ResponsiveDrawer>;
}

interface SiteNavDrawerProps {
  children: React.ReactNode;
}
function DesktopDrawer({ children }: SiteNavDrawerProps) {
  return (
    <Drawer
      variant="permanent"
      anchor="left"
      sx={{
        width: NAV_WIDTH,
        flexShrink: 0,
        [`& .MuiDrawer-paper`]: { width: NAV_WIDTH, boxSizing: "border-box" },
      }}
    >
      <Toolbar variant="dense" />
      <Box sx={{ display: "flex", flexDirection: "column", gap: NAV_GAP, alignItems: "center", paddingTop: NAV_GAP }}>
        {children}
      </Box>
    </Drawer>
  );
}

function MobileDrawer({ children }: SiteNavDrawerProps) {
  return (
    <Drawer
      variant="permanent"
      anchor="bottom"
      sx={{
        [`& .MuiDrawer-paper`]: {
          height: NAV_HEIGHT,
          boxSizing: "border-box",
          display: "flex",
          justifyContent: "center",
        },
      }}
    >
      <Box
        sx={{
          display: "flex",
          gap: NAV_GAP,
          alignItems: "center",
          justifyContent: "space-evenly",
        }}
      >
        {children}
      </Box>
    </Drawer>
  );
}

function LeftNavItemSkeletons() {
  return (
    <>
      <Skeleton variant="rounded" height={NAV_ITEM_WIDTH} width={NAV_ITEM_WIDTH} />
      <Skeleton variant="rounded" height={NAV_ITEM_WIDTH} width={NAV_ITEM_WIDTH} />
      <Skeleton variant="rounded" height={NAV_ITEM_WIDTH} width={NAV_ITEM_WIDTH} />
      <Skeleton variant="rounded" height={NAV_ITEM_WIDTH} width={NAV_ITEM_WIDTH} />
    </>
  );
}
