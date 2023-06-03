import { Skeleton, Box } from "@mui/material";

export function IncidentDetailsDataPanelSkeleton() {
  return (
    <Box display="flex" flexDirection="column" gap={2}>
      <Skeleton variant="rounded" height="20px" width="70%" />
      <Skeleton variant="rounded" height="12px" width="40%" />
      <Skeleton variant="rounded" sx={{ paddingBottom: "60%" }} width="100%" />
      <Box display="flex" flexDirection={{ xs: "column", sm: "row" }} width="100%" gap={2}>
        <Box flex="1" display="flex" flexDirection="column" gap={2}>
          <Skeleton variant="rounded" height="12px" width="40%" />
          <Skeleton variant="rounded" height="12px" width="60%" />
          <Skeleton variant="rounded" height="12px" width="50%" />
        </Box>
        <Box flex="1" display="flex" flexDirection="column" gap={2}>
          <Skeleton variant="rounded" height="40px" width="100%" />
          <Skeleton variant="rounded" height="40px" width="100%" />
          <Skeleton variant="rounded" height="40px" width="100%" />
        </Box>
      </Box>
    </Box>
  );
}
