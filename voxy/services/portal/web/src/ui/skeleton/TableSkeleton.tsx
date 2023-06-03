import { Box, Paper, Skeleton } from "@mui/material";
import { getRandomInt } from "shared/utilities/random";

interface TableSkeletonProps {
  padding?: number | string;
}
export function TableSkeleton({ padding = 3 }: TableSkeletonProps) {
  return (
    <Paper>
      <Box sx={{ display: "flex", padding, gap: 4, width: "100%", flexDirection: "column" }}>
        <Box>
          <Skeleton variant="rounded" height={14} width="60%" sx={{ marginBottom: "4px" }} />
          <Skeleton variant="rounded" height={14} width="80%" />
        </Box>
        {[...Array(10)].map((_, i) => {
          return (
            <Box key={i} sx={{ display: "flex", justifyContent: "space-between" }}>
              <Skeleton variant="rounded" height={12} width={`${getRandomInt(60, 200)}px`} />
              <Skeleton variant="rounded" height={12} width="32px" />
            </Box>
          );
        })}
      </Box>
    </Paper>
  );
}
