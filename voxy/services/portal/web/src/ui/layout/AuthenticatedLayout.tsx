import { Toolbar, Box, useTheme, useMediaQuery } from "@mui/material";
import { INTERNAL_SITE_ACCESS, Permission, TOOLBOX, useCurrentUser } from "features/auth";
import { Toolbox } from "features/toolbox";
import { TopNav } from "./TopNav";
import { useLocation } from "react-router-dom";
import { useCallback, useEffect, useMemo } from "react";
import { GenericErrorPage } from "features/errors";
import { CurrentAnnouncements } from "ui/Announcements";
import { Toaster } from "react-hot-toast";
import { Transition } from "@headlessui/react";
import { BackgroundSpinner } from "ui/spinners";
import { RedirectToSite } from "features/common";
import { NAV_WIDTH } from "./constants";

interface AuthenticatedLayoutProps {
  children: React.ReactNode;
  contextualNav?: React.ReactNode;
  requiredPermission?: Permission;
  hideSiteSelector?: boolean;
  mobileNavOnly?: boolean;
}
export function AuthenticatedLayout({
  children,
  contextualNav,
  requiredPermission,
  hideSiteSelector,
  mobileNavOnly,
}: AuthenticatedLayoutProps) {
  const { currentUser, isAuthenticated, isLoading, loginWithRedirect } = useCurrentUser();
  const hasPermission = !requiredPermission || currentUser?.hasGlobalPermission(requiredPermission);

  const theme = useTheme();
  const mdBreakpoint = useMediaQuery(theme.breakpoints.up("md"));
  const location = useLocation();
  const noTopNav = location.pathname.startsWith("/account") || location.pathname === "support";

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      loginWithRedirect();
    } else {
      if (!currentUser) {
        return;
      }
    }
  }, [isLoading, isAuthenticated, loginWithRedirect, currentUser]);

  const showSpinner = useMemo(() => {
    return isLoading || !currentUser || !isAuthenticated;
  }, [isLoading, currentUser, isAuthenticated]);

  const sitePermissionCheck = useCallback(() => {
    if (currentUser?.sites?.length || currentUser?.hasGlobalPermission(INTERNAL_SITE_ACCESS)) {
      return children;
    }

    return (
      <GenericErrorPage
        title="Access Denied - Voxel"
        message={`Your account is not configured correctly. Contact an Admin about this issue.`}
      />
    );
  }, [currentUser, children]);

  return (
    <>
      <Transition
        show={showSpinner}
        enter="transition-opacity duration-150"
        enterFrom="opacity-0"
        enterTo="opacity-100"
        leave="transition-opacity duration-150"
        leaveFrom="opacity-100"
        leaveTo="opacity-0"
      >
        <BackgroundSpinner />
      </Transition>
      {!showSpinner ? (
        <RedirectToSite>
          <Box sx={{ display: "flex" }}>
            {!noTopNav && <TopNav hideSiteSelector={hideSiteSelector} />}
            {contextualNav}
            {/* TODO(hq): sample mui code for permanent drawer wasnt working well for a few pages, find better way to fill available width? */}
            <Box
              component="main"
              sx={{ flexGrow: 1, width: `calc(100% - ${mdBreakpoint && !mobileNavOnly ? NAV_WIDTH : "0px"})` }}
            >
              <Toolbar variant="dense" />
              {hasPermission ? (
                sitePermissionCheck()
              ) : (
                <GenericErrorPage
                  title="Access Denied - Voxel"
                  message={`It looks like you don't have access to this page. Check the URL and try again.`}
                />
              )}
              {!mdBreakpoint && <Toolbar variant="dense" />}
              {currentUser?.hasGlobalPermission(TOOLBOX) ? <Toolbox /> : null}
              <CurrentAnnouncements />
              <Toaster position="bottom-center" />
            </Box>
          </Box>
        </RedirectToSite>
      ) : null}
    </>
  );
}
