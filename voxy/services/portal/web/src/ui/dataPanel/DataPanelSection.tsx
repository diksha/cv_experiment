import { ReactNode } from "react";
import { CircularProgress, Typography, Box, useTheme } from "@mui/material";

interface DataPanelSectionProps {
  children: ReactNode;
  title?: ReactNode;
  loading?: boolean;
  bodyPadding?: number;
  sortButton?: ReactNode;
}
export function DataPanelSection({ children, title, loading, bodyPadding, sortButton }: DataPanelSectionProps) {
  const theme = useTheme();
  const borderColor = theme.palette.grey[300];
  const borderWidth = "1px";
  bodyPadding = typeof bodyPadding === "number" ? bodyPadding : 2;

  // If title is a string, format it
  if (typeof title === "string") {
    title = (
      <Typography variant="h4" fontWeight="bold">
        {title}
      </Typography>
    );
  }

  return (
    <Box borderRadius="8px" sx={{ borderWidth, borderColor }}>
      {title ? (
        <Box
          padding={2}
          sx={{
            borderBottomWidth: borderWidth,
            borderBottomColor: borderColor,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div>{title}</div>
          {sortButton}
        </Box>
      ) : null}
      {loading ? (
        <Box textAlign="center" padding={4}>
          <CircularProgress sx={{ color: borderColor }} />
        </Box>
      ) : (
        <Box padding={bodyPadding}>{children}</Box>
      )}
    </Box>
  );
}
