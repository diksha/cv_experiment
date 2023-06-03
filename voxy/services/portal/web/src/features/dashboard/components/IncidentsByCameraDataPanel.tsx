import { DateTime } from "luxon";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useLazyQuery } from "@apollo/client";
import { getNodes } from "graphql/utils";
import { DateRange, DataPanel, DataPanelSection, DateRangePicker } from "ui";
import { GET_DATA_PANEL_INCIDENTS } from "features/incidents";
import { IncidentTrendsMapping, IncidentType } from "features/dashboard";
import { DataPanelIncidentList } from "features/incidents/components/dataPanels";
import {
  GetDataPanelIncidents,
  GetDataPanelIncidentsVariables,
  GetDataPanelIncidents_currentUser_site_incidents_edges_node,
} from "__generated__/GetDataPanelIncidents";
import { FilterBag, serializeFilters } from "shared/filter";
import { filterNullValues } from "shared/utilities/types";
import { MAX_HR_DAYS } from "./constants";
import { TimeBucketWidth } from "__generated__/globalTypes";
import { fillIncidentAggregateGroups, groupTotalsByDate } from "./trends/utils";
import { Box, Typography } from "@mui/material";
import { CalendarToday } from "@mui/icons-material";
import { IncidentsByCameraBarChart } from "./IncidentsByCameraBarChart";

interface IncidentListItem extends GetDataPanelIncidents_currentUser_site_incidents_edges_node {}

interface IncidentsByCameraDataPanelProps {
  cameraId: string;
  cameraName: string;
  incidentTypes: IncidentType[];
  open: boolean;
  startDate: DateTime;
  endDate: DateTime;
  timezone: string;
  onClose: () => void;
}
export function IncidentsByCameraDataPanel({
  cameraId,
  cameraName,
  incidentTypes,
  open,
  startDate,
  endDate,
  timezone,
  onClose,
}: IncidentsByCameraDataPanelProps) {
  const [dateRangeFilter, setDateRangeFilter] = useState<DateRange>({
    startDate: startDate,
    endDate: endDate,
  });
  const [backDateRangeFilter, setBackDateRangeFilter] = useState<DateRange>({
    startDate: null,
    endDate: null,
  });
  const [fetchingMore, setFetchingMore] = useState(false);
  const [filteredIncidentTypes, setFilteredIncidentTypes] = useState<string[]>([]);

  const filterBag: FilterBag = {
    CAMERA: { value: [cameraId] },
    INCIDENT_TYPE: { value: [...filteredIncidentTypes] },
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

  // TODO(hq): find out why DashboardV3 re-renders after changing date with PRODUCTION_LINE_DOWN(dev env data) selected

  useEffect(() => {
    setFilteredIncidentTypes(incidentTypes.map((i) => i.key));
  }, [incidentTypes.length]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (open) {
      getDataPanelIncidents();
    }
  }, [open, getDataPanelIncidents, filteredIncidentTypes, cameraId]);

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
  };

  const handleOnBack = () => {
    setDateRangeFilter(backDateRangeFilter);
    setBackDateRangeFilter({ startDate: null, endDate: null });
  };

  const handleClose = () => {
    setDateRangeFilter({ startDate, endDate });
    setBackDateRangeFilter({ startDate: null, endDate: null });
    setFilteredIncidentTypes(incidentTypes.map((i) => i.key));
    onClose();
  };

  const onFilter1Day = useCallback(
    (dateRange: DateRange) => {
      setDateRangeFilter(dateRange);
      setBackDateRangeFilter(dateRangeFilter);
    },
    [dateRangeFilter]
  );

  const stackedBarChartByCamera = useMemo(() => {
    if (!dateRangeFilter.startDate || !dateRangeFilter.endDate) {
      return <></>;
    }
    const groups = data?.currentUser?.site?.incidentAnalytics?.incidentAggregateGroups || [];
    const mapping: IncidentTrendsMapping = fillIncidentAggregateGroups(
      groups,
      incidentTypes,
      groupBy,
      dateRangeFilter.startDate,
      dateRangeFilter.endDate,
      timezone
    );
    const totals = groupTotalsByDate(mapping);
    return (
      <IncidentsByCameraBarChart
        data={totals}
        timezone={timezone}
        groupBy={groupBy}
        incidentTypes={incidentTypes}
        filteredIncidentTypes={filteredIncidentTypes}
        setFilteredIncidentTypes={setFilteredIncidentTypes}
        onFilter1Day={onFilter1Day}
      />
    );
  }, [
    data,
    timezone,
    dateRangeFilter.startDate,
    dateRangeFilter.endDate,
    incidentTypes,
    filteredIncidentTypes,
    groupBy,
    onFilter1Day,
  ]);

  return (
    <DataPanel open={open} onClose={handleClose}>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <Box sx={{ display: "flex", alignItems: "center" }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h2">{cameraName}</Typography>
          </Box>
          <Box>
            <DateRangePicker
              uiKey="button-filter-by-date-range"
              onChange={handleDateRangeChange}
              onBack={handleOnBack}
              values={dateRangeFilter}
              backValues={backDateRangeFilter}
              icon={<CalendarToday sx={{ height: 16, width: 16 }} />}
              timezone={timezone}
            />
          </Box>
        </Box>
        <DataPanelSection>{stackedBarChartByCamera}</DataPanelSection>
        <DataPanelIncidentList
          loading={loading}
          incidents={incidents}
          title="Most Recent Incidents"
          showLoadMore={showLoadMore}
          handleLoadMore={handleLoadMore}
          fetchingMore={fetchingMore}
          uiKey="camera-details-data-panel"
        />
      </Box>
    </DataPanel>
  );
}
