import Toolbar from "@mui/material/Toolbar";
import { AppBar, useTheme, Box, useMediaQuery } from "@mui/material";
import { Link } from "react-router-dom";
import { LogoIcon } from "ui";
import { SiteSelector } from "features/organizations";
import { NAV_ITEM_WIDTH, NAV_ITEM_ICON_WIDTH, NAV_WIDTH } from "./constants";
import { AccountMenu } from "./AccountMenu";

interface TopNavProps {
  hideSiteSelector?: boolean;
}
export function TopNav({ hideSiteSelector }: TopNavProps) {
  const theme = useTheme();
  const mdBreakpoint = useMediaQuery(theme.breakpoints.up("md"));

  return (
    <AppBar
      position="fixed"
      sx={{
        boxShadow: "none",
        borderBottomWidth: 1,
        borderBottomColor: theme.palette.grey[300],
        backgroundColor: theme.palette.grey[100],
        zIndex: theme.zIndex.drawer + 1,
      }}
    >
      <Toolbar disableGutters variant="dense" sx={{ display: "flex", paddingX: 0, paddingY: 1 }}>
        <Box
          sx={{
            display: "flex",
            justifyContent: "center",
            width: NAV_WIDTH,
          }}
        >
          <Box
            component={Link}
            to="/"
            data-ui-key="top-nav-logo-link"
            sx={{
              display: "flex",
              justifyContent: "center",
              width: NAV_ITEM_WIDTH,
              height: NAV_ITEM_WIDTH,
              borderRadius: 2,
              transition: "transform 0.1s ease-out",
              "&:hover": { transform: "scale(1.15)" },
            }}
          >
            <Box sx={{ width: NAV_ITEM_ICON_WIDTH }}>
              <LogoIcon variant="dark" />
            </Box>
          </Box>
        </Box>
        <Box
          sx={{
            flexGrow: 1,
            display: "flex",
            alignItems: "center",
            gap: 2,
            paddingRight: 2,
            justifyContent: hideSiteSelector ? "flex-end" : "flex-start",
          }}
        >
          {!hideSiteSelector && (
            <Box sx={{ flexGrow: 1 }}>
              <SiteSelector />
            </Box>
          )}
          {mdBreakpoint && <AccountMenu />}
        </Box>
      </Toolbar>
    </AppBar>
  );
}
