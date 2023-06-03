import { Box, Paper, Skeleton } from "@mui/material";

interface VoxelScoreCardSkeletonProps {
  showOverallScore?: boolean;
  showHeader?: boolean;
  minWidth?: number | string;
  padding?: number | string;
}
export function VoxelScoreCardSkeleton({
  showOverallScore,
  showHeader,
  minWidth,
  padding = 3,
}: VoxelScoreCardSkeletonProps) {
  return (
    <Paper sx={{ minWidth }}>
      <Box sx={{ display: "flex", padding, gap: 4, width: "100%", flexDirection: "column" }}>
        {showOverallScore && <Skeleton variant="circular" width={80} height={80} sx={{ margin: "0 auto" }} />}
        {showHeader && (
          <Box>
            <Skeleton variant="rounded" height={14} width="60%" sx={{ marginBottom: "4px" }} />
            <Skeleton variant="rounded" height={14} width="80%" />
          </Box>
        )}
        <Skeleton variant="rounded" width="40%" height={12} />
        <Skeleton variant="rounded" width="60%" height={12} />
        <Skeleton variant="rounded" width="90%" height={12} />
        <Skeleton variant="rounded" width="50%" height={12} />
        <Skeleton variant="rounded" width="80%" height={12} />
        <Skeleton variant="rounded" width="70%" height={12} />
      </Box>
    </Paper>
  );
}
