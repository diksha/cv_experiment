import { Skeleton, Box, Paper } from "@mui/material";

export function StatisticCardSkeleton() {
  return (
    <Paper>
      <Box display="flex" flexDirection="column" gap={2} p={2} width="100%">
        <Box>
          <Skeleton variant="rounded" height={14} width="60%" />
        </Box>
        <Box display="flex" gap={4}>
          <Skeleton variant="rounded" height={60} width={50} />
          <Skeleton variant="rounded" height={60} width={50} />
        </Box>
      </Box>
    </Paper>
  );
}
