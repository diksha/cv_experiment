import { Box, Paper, Skeleton } from "@mui/material";
import { getRandomInt } from "shared/utilities/random";

export function DowntimeBarChartSkeleton() {
  return (
    <Paper>
      <Box sx={{ display: "flex", padding: 3, gap: 2, width: "100%", flexDirection: "column" }}>
        <Skeleton variant="rounded" width="40%" height={12} />
        <Skeleton variant="rounded" width="30%" height={12} />
        <Box sx={{ display: "flex", flexDirection: "column", width: "100%" }}>
          <Box sx={{ display: "flex", alignItems: "flex-end", gap: 0.5 }}>
            {[...Array(28)].map((_, index) => (
              <Skeleton key={`skeleton-bar-${index}`} variant="rounded" width="10px" height={getRandomInt(10, 40)} />
            ))}
          </Box>
        </Box>
        <Skeleton variant="rounded" width="100%" height={20} />
      </Box>
    </Paper>
  );
}
