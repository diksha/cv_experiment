import { DateTime } from "luxon";
import { DataPanel, DataPanelSection } from "ui";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Box, FormControl, InputLabel, MenuItem, Select, SelectChangeEvent, Typography } from "@mui/material";
import { DateRange, DateRangePicker } from "ui";
import { TotalsByDate, IncidentTrend, IncidentTrendsMapping } from "./types";
import { CalendarToday } from "@mui/icons-material";
import { IncidentTrendBarChart } from "./IncidentTrendBarChart";
import { DataPanelIncidentList } from "features/incidents/components/dataPanels";
import { FilterBag, serializeFilters } from "shared/filter";
import { useLazyQuery } from "@apollo/client";
import { getNodes } from "graphql/utils";
import { filterNullValues } from "shared/utilities/types";
import { GET_DATA_PANEL_INCIDENTS } from "features/incidents";
import { IncidentType, Camera, MAX_HR_DAYS } from "features/dashboard";
import { fillIncidentAggregateGroups, fillIncidentAggregateGroupsByCamera } from "./utils";
import {
  GetDataPanelIncidents,
  GetDataPanelIncidentsVariables,
  GetDataPanelIncidents_currentUser_site_incidents_edges_node,
} from "__generated__/GetDataPanelIncidents";
import { TimeBucketWidth } from "__generated__/globalTypes";
import { IncidentTrendByCameraBarChart } from "./IncidentTrendByCameraBarChart";
import { analytics } from "shared/utilities/analytics";

interface IncidentListItem extends GetDataPanelIncidents_currentUser_site_incidents_edges_node {}

interface IncidentTrendDataPanelProps {
  trend: IncidentTrend;
  open: boolean;
  startDate: DateTime;
  endDate: DateTime;
  timezone: string;
  cameras: Camera[];
  onClose: () => void;
}
export function IncidentTrendDataPanel({
  trend,
  open,
  startDate,
  endDate,
  timezone,
  cameras,
  onClose,
}: IncidentTrendDataPanelProps) {
  const [dateRangeFilter, setDateRangeFilter] = useState<DateRange>({
    startDate: startDate,
    endDate: endDate,
  });
  const [backDateRangeFilter, setBackDateRangeFilter] = useState<DateRange>({
    startDate: null,
    endDate: null,
  });
  const [fetchingMore, setFetchingMore] = useState(false);
  const [chartType, setChartType] = useState<string>("totalIncidents");
  const [filteredCameras, setFilteredCameras] = useState(cameras.map((c) => c.id));

  // TODO: special check for __ALL__ constant
  const filterBag: FilterBag = {
    INCIDENT_TYPE: { value: [trend.incidentTypeKey] },
    CAMERA: { value: [...filteredCameras] },
  };
  const serializedFilters = serializeFilters(filterBag || {});

  const daysDiff =
    dateRangeFilter.startDate && dateRangeFilter.endDate
      ? dateRangeFilter.endDate.diff(dateRangeFilter.startDate, ["days"])
      : { days: 0 };
  const groupBy = daysDiff.days > MAX_HR_DAYS ? TimeBucketWidth.DAY : TimeBucketWidth.HOUR;

  const [getDataPanelIncidents, { data, loading, fetchMore }] = useLazyQuery<
    GetDataPanelIncidents,
    GetDataPanelIncidentsVariables
  >(GET_DATA_PANEL_INCIDENTS, {
    fetchPolicy: "network-only",
    variables: {
      first: 5,
      startTimestamp: dateRangeFilter.startDate ? dateRangeFilter.startDate.startOf("day").toISO() : null,
      endTimestamp: dateRangeFilter.endDate ? dateRangeFilter.endDate.endOf("day").toISO() : null,
      filters: serializedFilters,
      groupBy,
    },
  });

  useEffect(() => {
    if (open) {
      getDataPanelIncidents();
    }
  }, [open, getDataPanelIncidents, filteredCameras]);

  const onFilter1Day = useCallback(
    (dateRange: DateRange) => {
      setDateRangeFilter(dateRange);
      setBackDateRangeFilter(dateRangeFilter);
    },
    [dateRangeFilter]
  );

  const barChart = useMemo(() => {
    if (!dateRangeFilter.startDate || !dateRangeFilter.endDate) {
      return <></>;
    }
    const groups = data?.currentUser?.site?.incidentAnalytics?.incidentAggregateGroups || [];
    const incidentType: IncidentType = {
      name: trend.name,
      key: trend.incidentTypeKey,
      backgroundColor: "",
    };
    const aggregateGroups: IncidentTrendsMapping = fillIncidentAggregateGroups(
      groups,
      [incidentType],
      groupBy,
      dateRangeFilter.startDate,
      dateRangeFilter.endDate,
      timezone,
      trend.incidentTypeKey,
      trend.name
    );
    return (
      <IncidentTrendBarChart
        data={aggregateGroups[trend.incidentTypeKey]}
        timezone={timezone}
        groupBy={groupBy}
        onFilter1Day={onFilter1Day}
      />
    );
  }, [
    data,
    timezone,
    dateRangeFilter.startDate,
    dateRangeFilter.endDate,
    groupBy,
    trend.incidentTypeKey,
    trend.name,
    onFilter1Day,
  ]);

  const stackedBarChartByCamera = useMemo(() => {
    if (!dateRangeFilter.startDate || !dateRangeFilter.endDate) {
      return <></>;
    }
    const groups = data?.currentUser?.site?.incidentAnalytics?.incidentAggregateGroups || [];
    const totals: TotalsByDate[] = fillIncidentAggregateGroupsByCamera(
      groups,
      cameras,
      groupBy,
      dateRangeFilter.startDate,
      dateRangeFilter.endDate,
      timezone,
      trend.incidentTypeKey,
      trend.name
    );
    return (
      <IncidentTrendByCameraBarChart
        data={totals}
        timezone={timezone}
        groupBy={groupBy}
        cameras={cameras}
        filteredCameraIds={filteredCameras}
        setFilteredCameras={setFilteredCameras}
        onFilter1Day={onFilter1Day}
      />
    );
  }, [
    data,
    timezone,
    dateRangeFilter.startDate,
    dateRangeFilter.endDate,
    groupBy,
    trend.incidentTypeKey,
    trend.name,
    cameras,
    filteredCameras,
    onFilter1Day,
  ]);

  const incidentNodes = getNodes<IncidentListItem>(data?.currentUser?.site?.incidents);
  const incidents = filterNullValues<IncidentListItem>(incidentNodes);
  const hasNextPage = data?.currentUser?.site?.incidents?.pageInfo?.hasNextPage;
  const endCursor = hasNextPage && data?.currentUser?.site?.incidents?.pageInfo?.endCursor;
  const showLoadMore = !!(hasNextPage && endCursor);

  const handleLoadMore = async () => {
    if (showLoadMore) {
      setFetchingMore(true);
      try {
        await fetchMore({
          variables: {
            first: 10,
            after: endCursor,
          },
        });
      } finally {
        setFetchingMore(false);
      }
    }
  };

  const handleDateRangeChange = (dateRange: DateRange) => {
    setDateRangeFilter(dateRange);
    setBackDateRangeFilter({ startDate: null, endDate: null });
    analytics.trackCustomEvent("changeDataPanelDateRange");
  };

  const handleOnBack = () => {
    setDateRangeFilter(backDateRangeFilter);
    setBackDateRangeFilter({ startDate: null, endDate: null });
  };

  const handleChartTypeChange = (event: SelectChangeEvent) => {
    setFilteredCameras(cameras.map((c) => c.id));
    setChartType(event.target.value);
  };

  return (
    <DataPanel open={open} onClose={onClose}>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }} data-ui-key="incident-trend-data-panel">
        <Box sx={{ display: "flex", alignItems: "center" }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h2">{trend.name}</Typography>
          </Box>
          <Box>
            <DateRangePicker
              uiKey="datapanel-date-range-change"
              onChange={handleDateRangeChange}
              onBack={handleOnBack}
              values={dateRangeFilter}
              backValues={backDateRangeFilter}
              icon={<CalendarToday sx={{ height: 16, width: 16 }} />}
              timezone={timezone}
            />
          </Box>
        </Box>
        <DataPanelSection>
          <FormControl size="small" sx={{ width: "200px", marginBottom: "20px" }}>
            <InputLabel id="chart-type-select-label">Type</InputLabel>
            <Select
              labelId="chart-type-select-label"
              id="chart-type-select"
              autoWidth
              value={chartType}
              label="Type"
              onChange={handleChartTypeChange}
            >
              <MenuItem value="totalIncidents">Total Incidents</MenuItem>
              <MenuItem value="incidentsByCamera">Incidents by Camera</MenuItem>
            </Select>
          </FormControl>
          {chartType === "totalIncidents" && barChart}
          {chartType === "incidentsByCamera" && stackedBarChartByCamera}
        </DataPanelSection>
        <DataPanelIncidentList
          loading={loading}
          incidents={incidents}
          title="Most Recent Incidents"
          showLoadMore={showLoadMore}
          handleLoadMore={handleLoadMore}
          fetchingMore={fetchingMore}
          uiKey="datapanel-activity-incident"
        />
      </Box>
    </DataPanel>
  );
}
