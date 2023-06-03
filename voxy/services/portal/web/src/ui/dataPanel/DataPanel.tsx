import { ReactNode } from "react";
import { useMediaQuery, Box, Drawer, useTheme } from "@mui/material";
import { CloseButton, CloseIconVariant, ActionButtonWrapper } from "./actionButtons";

interface DataPanelProps {
  open: boolean;
  children: ReactNode;
  title?: string;
  onClose: () => void;
  closeIconVariant?: CloseIconVariant;
}

export function DataPanel({ open, children, onClose, closeIconVariant }: DataPanelProps) {
  const theme = useTheme();
  const xsBreakpoint = useMediaQuery(theme.breakpoints.down("sm"));
  return (
    <Drawer
      anchor={xsBreakpoint ? "bottom" : "right"}
      variant="temporary"
      open={open}
      onClose={onClose}
      sx={{
        // This zIndex value keeps the data panel above app bar + other drawers
        zIndex: theme.zIndex.drawer + 2,
      }}
      PaperProps={{
        sx: {
          display: "block",
          width: xsBreakpoint ? "100%" : "600px",
          maxWidth: xsBreakpoint ? "100%" : "85%",
          height: xsBreakpoint ? "90%" : "100%",
          borderTopLeftRadius: xsBreakpoint ? "16px" : 0,
          borderTopRightRadius: xsBreakpoint ? "16px" : 0,
        },
      }}
    >
      <ActionButtonWrapper>
        <CloseButton onClick={onClose} variant={closeIconVariant} />
      </ActionButtonWrapper>
      <Box
        sx={{
          position: "relative",
        }}
      >
        <Box padding="1rem">{children}</Box>
      </Box>
    </Drawer>
  );
}
