import { Box, Paper, Skeleton } from "@mui/material";
import { getRandomInt } from "shared/utilities/random";

export function IncidentTrendSkeleton() {
  return (
    <Paper>
      <Box sx={{ display: "flex", padding: 3, gap: 2, width: "100%" }}>
        <Box sx={{ display: "flex", flexDirection: "column", gap: 2, width: "100%" }}>
          <Skeleton variant="rounded" width="40%" height={20} />
          <Skeleton variant="rounded" width="30%" height={40} />
          <Skeleton variant="rounded" width="50%" height={20} />
        </Box>
        <Box sx={{ display: "flex", flexDirection: "column", width: "100%" }}>
          <Box sx={{ display: "flex", justifyContent: "flex-end", flex: 1 }}>
            <Skeleton variant="rounded" width="20%" height={20} />
          </Box>
          <Box sx={{ display: "flex", justifyContent: "flex-end", alignItems: "flex-end", gap: 0.5 }}>
            {[...Array(20)].map((_, index) => (
              <Skeleton key={`skeleton-bar-${index}`} variant="rounded" width="10px" height={getRandomInt(1, 30)} />
            ))}
          </Box>
        </Box>
      </Box>
    </Paper>
  );
}
