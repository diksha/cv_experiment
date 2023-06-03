import { DowntimeBarChartSkeleton, DataPanel, DataPanelSection, DateRange, DateRangePicker } from "ui";
import { useState, useMemo, useEffect } from "react";
import { DateTime } from "luxon";
import { TimeBucketWidth } from "shared/enums";
import { DowntimeChartProps } from "./DowntimeChart";
import { DowntimeEventList } from "./DowntimeEventList";
import { DowntimeCardChartWrapper } from "./DowntimeCardChartWrapper";
import { DowntimeCardChartWrapperSimple } from "./DowntimeCardChartWrapperSimple";
import { ChartBrushRange, ProductionLineStatusDataPoint } from "./types";
import { Box, Stack, Typography, useTheme } from "@mui/material";
import { CalendarToday } from "@mui/icons-material";
import { useLazyQuery } from "@apollo/client";
import { GET_PRODUCTION_LINES } from "./queries";
import { GetProductionLines, GetProductionLinesVariables } from "__generated__/GetProductionLines";

interface DowntimeCardProps extends DowntimeChartProps {
  productionLineId: string;
  title: string;
  timeBucketWidth: TimeBucketWidth;
  data: ProductionLineStatusDataPoint[];
  timezone: string;
}

export function DowntimeCard({ productionLineId, title, timeBucketWidth, data, timezone }: DowntimeCardProps) {
  const theme = useTheme();
  const [dataPanelOpen, setDataPanelOpen] = useState(false);
  const [brushRange, setBrushRange] = useState<ChartBrushRange | undefined>();
  const [selectedBarIndex, setSelectedBarIndex] = useState<number | null>(null);
  const yesterdayStart = DateTime.now().setZone(timezone).minus({ days: 1 }).startOf("day");
  const yesterdayEnd = DateTime.now().setZone(timezone).minus({ days: 1 }).endOf("day");
  const [dateRangeFilter, setDateRangeFilter] = useState<DateRange>({
    startDate: yesterdayStart,
    endDate: yesterdayEnd,
  });

  const [getProductionLines, { data: filteredData, loading }] = useLazyQuery<
    GetProductionLines,
    GetProductionLinesVariables
  >(GET_PRODUCTION_LINES, {
    fetchPolicy: "network-only",
    variables: {
      startTimestamp: dateRangeFilter.startDate ? dateRangeFilter.startDate.startOf("day").toISO() : null,
      endTimestamp: dateRangeFilter.endDate ? dateRangeFilter.endDate.endOf("day").toISO() : null,
    },
  });

  useEffect(() => {
    if (dataPanelOpen) {
      getProductionLines();
    }
  }, [dataPanelOpen, getProductionLines]);

  const filteredStatus1h = useMemo(() => {
    const prodLine = filteredData?.currentUser?.site?.productionLines?.find((line) => line.id === productionLineId);
    return prodLine?.status1hGroups || data;
  }, [filteredData, data, productionLineId]);

  const selectedDataPoint: ProductionLineStatusDataPoint | undefined = useMemo(() => {
    const points = filteredStatus1h || data;
    if (typeof selectedBarIndex == "number" && points.length > selectedBarIndex) {
      return points[selectedBarIndex];
    }
  }, [filteredStatus1h, data, selectedBarIndex]);

  const [selectedBarStartTimestamp, selectedBarEndTimestamp] = useMemo(() => {
    const invalidRange = [undefined, undefined];

    if (!selectedDataPoint) {
      return invalidRange;
    }

    const startTimestamp = DateTime.fromISO(selectedDataPoint.dimensions.datetime);
    if (!startTimestamp.isValid) {
      return invalidRange;
    }

    let endTimestamp: DateTime;

    switch (timeBucketWidth) {
      case TimeBucketWidth.Day:
        endTimestamp = startTimestamp.plus({ days: 1 }).minus({ milliseconds: 1 });
        break;
      case TimeBucketWidth.Hour:
        endTimestamp = startTimestamp.plus({ hours: 1 }).minus({ milliseconds: 1 });
        break;
      default:
        return invalidRange;
    }
    return [startTimestamp, endTimestamp];
  }, [selectedDataPoint, timeBucketWidth]);

  const showEventList = selectedBarStartTimestamp && selectedBarEndTimestamp;

  const handleDataPanelOpen = () => {
    setDataPanelOpen(true);
  };

  const handleDrawerClose = () => {
    setDataPanelOpen(false);
  };

  const handleBarClick = (index: number) => {
    // When the brush range changes, the bar indexes change to reflect only the visible bars.
    // So the bar click event returns the selected bar index based on the visible bars, not
    // the underlying data array. We need to base our selectedBarIndex on the index
    // of the underlying data array, so we add the brush range start index (if any)
    // with the selected bar index:
    //
    //     range: 2:6
    //      data: 0 1 2 3 4 5 6 7 8 9
    //     brush:     0 1 2 3 4
    //
    //
    // If the user clicks the bar at data index 4, the bar click event will return index 2,
    // so if we add the start index + clicked index (2 + 2) we'll get the index of the
    // item in the data array.
    const newIndex = brushRange ? brushRange.startIndex + index : index;
    setSelectedBarIndex(newIndex);
  };

  const handleBrushChange = (range?: ChartBrushRange) => {
    // Reset selected bar index whenenever the brush changes
    setBrushRange(range);
    setSelectedBarIndex(null);
  };

  const handleDateRangeChange = (dateRange: DateRange) => {
    setDateRangeFilter(dateRange);
    setBrushRange(undefined);
    getProductionLines();
  };

  return (
    <>
      <DowntimeCardChartWrapperSimple
        title={title}
        data={data}
        onDataPanelOpen={handleDataPanelOpen}
        timezone={timezone}
      />
      <DataPanel open={dataPanelOpen} onClose={handleDrawerClose}>
        <Stack spacing={2} data-ui-key="downtime-card-data-panel">
          <Box sx={{ display: "flex", alignItems: "center" }}>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h2" fontWeight={800}>
                Downtime
              </Typography>
            </Box>
            <Box>
              <DateRangePicker
                uiKey="downtime-card-data-panel-daterange-filter"
                onChange={handleDateRangeChange}
                values={dateRangeFilter}
                icon={<CalendarToday sx={{ height: 16, width: 16 }} />}
                timezone={timezone}
              />
            </Box>
          </Box>
          <DataPanelSection>
            {loading ? (
              <DowntimeBarChartSkeleton />
            ) : (
              <DowntimeCardChartWrapper
                title={title}
                data={filteredStatus1h}
                showBrush
                brushRange={brushRange}
                onBarClick={handleBarClick}
                onBrushChange={handleBrushChange}
                timezone={timezone}
                greyscale={true}
              />
            )}
          </DataPanelSection>
          {showEventList ? (
            <DowntimeEventList
              productionLineId={productionLineId}
              startTimestamp={selectedBarStartTimestamp}
              endTimestamp={selectedBarEndTimestamp}
              timezone={timezone}
            />
          ) : (
            <DataPanelSection>
              <Typography textAlign="center" color={theme.palette.grey[500]}>
                Click or tap a bar on the chart above to view related events
              </Typography>
            </DataPanelSection>
          )}
        </Stack>
      </DataPanel>
    </>
  );
}
