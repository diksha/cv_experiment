import { DowntimeChart } from "./DowntimeChart";
import { ChevronRight } from "@mui/icons-material";
import { ChartBrushRange, ProductionLineStatusDataPoint } from "./types";
import { DowntimeSummary } from "./DowntimeSummary";
import { Stack, IconButton, Box, useTheme } from "@mui/material";
import { Card } from "ui";

interface DowntimeCardChartWrapperProps {
  title: string;
  data: ProductionLineStatusDataPoint[];
  showBrush?: boolean;
  brushRange?: ChartBrushRange;
  timezone: string;
  greyscale?: boolean;
  onBrushChange: (range: ChartBrushRange) => void;
  onDataPanelOpen?: () => void;
  onBarClick?: (index: number) => void;
}

export function DowntimeCardChartWrapper({
  title,
  data,
  showBrush,
  brushRange,
  timezone,
  greyscale,
  onDataPanelOpen,
  onBarClick,
  onBrushChange,
}: DowntimeCardChartWrapperProps) {
  const theme = useTheme();

  return (
    <Card>
      <Stack spacing={2}>
        <Box sx={{ display: "flex", gap: "1rem", alignItems: "center" }}>
          <Box sx={{ flex: 1 }}>
            <DowntimeSummary title={title} data={data} brushRange={brushRange} />
          </Box>
          {onDataPanelOpen ? (
            <div>
              <IconButton
                onClick={onDataPanelOpen}
                color="primary"
                sx={{
                  height: "40px",
                  width: "40px",
                  backgroundColor: theme.palette.grey[200],
                  "&:hover": {
                    backgroundColor: theme.palette.grey[300],
                  },
                  borderRadius: "8px",
                }}
              >
                <ChevronRight />
              </IconButton>
            </div>
          ) : null}
        </Box>
        <div className="basis-full md:basis-2/3">
          <div className="flex-1">
            <div className="pt-6 md:pt-0">
              <DowntimeChart
                data={data}
                showBrush={showBrush}
                brushRange={brushRange}
                onBrushChange={onBrushChange}
                onBarClick={onBarClick}
                timezone={timezone}
                greyscale={greyscale}
              />
            </div>
          </div>
        </div>
      </Stack>
    </Card>
  );
}
