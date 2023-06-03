import { DowntimeChartSimple } from "./DowntimeChartSimple";
import { ChartBrushRange, ProductionLineStatusDataPoint } from "./types";
import { DowntimeText } from "./DowntimeSummary";
import { Box, useTheme, Paper, Typography } from "@mui/material";
import { DateTime } from "luxon";
import { useMemo } from "react";

interface DowntimeCardChartWrapperSimpleProps {
  title: string;
  data: ProductionLineStatusDataPoint[];
  brushRange?: ChartBrushRange;
  timezone: string;
  onDataPanelOpen?: () => void;
}

export function DowntimeCardChartWrapperSimple({
  title,
  data,
  brushRange,
  timezone,
  onDataPanelOpen,
}: DowntimeCardChartWrapperSimpleProps) {
  const theme = useTheme();

  const prevDayData = useMemo(() => {
    const yesterdayStart = DateTime.now().setZone(timezone).minus({ days: 1 }).startOf("day");
    const yesterdayEnd = DateTime.now().setZone(timezone).minus({ days: 1 }).endOf("day");
    return data.filter((d) => {
      const date = DateTime.fromISO(d.dimensions.datetime);
      return date >= yesterdayStart && date <= yesterdayEnd;
    });
  }, [data, timezone]);
  return (
    <Paper
      onClick={onDataPanelOpen}
      sx={{ width: "100%", cursor: "pointer", ":hover": onDataPanelOpen ? { boxShadow: 1 } : null }}
      data-ui-key="downtime-card"
    >
      <Box sx={{ padding: 3, display: { xs: "block", sm: "flex" } }}>
        <Box sx={{ width: { xs: "100%", sm: "50%" } }}>
          <Box display="flex" justifyContent="space-between">
            <Typography variant="h4" sx={{ marginBottom: theme.spacing(1) }}>
              Downtime
            </Typography>
          </Box>
          <Typography fontSize={40} fontWeight={800} lineHeight={1.12}>
            <DowntimeText title={title} data={prevDayData} brushRange={brushRange} />
          </Typography>
          <Typography sx={{ color: theme.palette.grey[500] }}>{title} - Yesterday</Typography>
        </Box>
        <Box sx={{ width: { xs: "100%", sm: "50%" }, paddingTop: "12px" }}>
          <DowntimeChartSimple data={prevDayData} />
        </Box>
      </Box>
    </Paper>
  );
}
