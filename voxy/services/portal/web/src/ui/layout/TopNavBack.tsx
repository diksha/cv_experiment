import Toolbar from "@mui/material/Toolbar";
import { AppBar, useTheme, Box, Typography, useMediaQuery } from "@mui/material";
import { ArrowBack } from "@mui/icons-material";
import { Link } from "react-router-dom";
import { AccountMenu } from "./AccountMenu";
import { LogoIcon } from "ui";
import { NAV_ITEM_ICON_WIDTH, NAV_ITEM_WIDTH } from "./constants";
import { useRouting } from "shared/hooks";

type IconVariant = "back" | "logo";
interface TopNavBackProps {
  mobTitle: string;
  mobTo?: string;
  icon?: IconVariant;
  children?: React.ReactNode;
}
export function TopNavBack({ mobTitle, mobTo, icon = "back", children }: TopNavBackProps) {
  const theme = useTheme();
  const { locationState, newLocationState } = useRouting();
  const mdBreakpoint = useMediaQuery(theme.breakpoints.up("md"));
  const title = mdBreakpoint ? "Settings" : mobTitle;
  const to = mdBreakpoint || !mobTo ? locationState?.visitedDash || "/" : mobTo;

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
      <Toolbar
        disableGutters
        variant="dense"
        sx={{ display: "flex", paddingX: 2, paddingY: 1, justifyContent: "space-between" }}
      >
        <Box sx={{ display: "flex", alignItems: "center" }} component={Link} to={to} state={newLocationState}>
          {icon === "back" && (
            <ArrowBack sx={{ width: "24px", height: "24px", color: theme.palette.primary.main, marginRight: "12px" }} />
          )}
          {icon === "logo" && (
            <Box
              sx={{
                display: "flex",
                justifyContent: "center",
                width: NAV_ITEM_WIDTH,
                height: NAV_ITEM_WIDTH,
                marginRight: "4px",
              }}
            >
              <Box sx={{ width: NAV_ITEM_ICON_WIDTH }}>
                <LogoIcon variant="dark" />
              </Box>
            </Box>
          )}
          <Typography variant="h4">{title}</Typography>
        </Box>
        {mdBreakpoint ? <AccountMenu /> : children}
      </Toolbar>
    </AppBar>
  );
}
