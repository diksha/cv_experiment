import { ReactNode } from "react";
import { useMediaQuery, Box, useTheme } from "@mui/material";
import { ACTION_BUTTON_OFFSET, ACTION_BUTTON_MARGIN } from "./constants";

interface ActionButtonWrapperProps {
  children: ReactNode;
}
export function ActionButtonWrapper({ children }: ActionButtonWrapperProps) {
  const theme = useTheme();
  const xsBreakpoint = useMediaQuery(theme.breakpoints.down("sm"));

  return (
    <Box
      position="fixed"
      display="flex"
      flexDirection={xsBreakpoint ? "row" : "column"}
      alignItems="center"
      gap={1}
      sx={{
        top: xsBreakpoint ? undefined : ACTION_BUTTON_OFFSET,
        right: xsBreakpoint ? ACTION_BUTTON_OFFSET : undefined,
        marginTop: xsBreakpoint ? ACTION_BUTTON_MARGIN : undefined,
        marginLeft: xsBreakpoint ? undefined : ACTION_BUTTON_MARGIN,
      }}
    >
      {children}
    </Box>
  );
}
