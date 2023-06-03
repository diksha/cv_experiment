import { DataPanelSection } from "ui";
import { Skeleton, Box, useTheme } from "@mui/material";

export function EventListSkeleton() {
  const theme = useTheme();
  const itemCount = 3;

  return (
    <DataPanelSection bodyPadding={0} title={<Skeleton variant="rounded" height="14px" width="40%" />}>
      {[...Array(itemCount)].map((_, index) => {
        const lastItem = index === itemCount - 1;
        const borderBottom = lastItem ? "" : `1px solid ${theme.palette.grey[300]}`;

        return (
          <Box key={`event-list-skeleton-${index}`} display="flex" gap={2} padding={2} borderBottom={borderBottom}>
            <Box>
              <Skeleton variant="rounded" height="60px" width="100px" />
            </Box>
            <Box display="flex" flexDirection="column" gap={2} flex="1">
              <Skeleton variant="rounded" height="20px" width="70%" />
              <Skeleton variant="rounded" height="12px" width="40%" />
            </Box>
          </Box>
        );
      })}
    </DataPanelSection>
  );
}
