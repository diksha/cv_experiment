import { DateTime } from "luxon";
import { useEffect, useState } from "react";
import { useLazyQuery } from "@apollo/client";
import { getNodes } from "graphql/utils";
import { DateRange, DataPanel, DateRangePicker } from "ui";
import { GET_DATA_PANEL_INCIDENTS } from "features/incidents";
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
import { Box, Typography } from "@mui/material";
import { CalendarToday } from "@mui/icons-material";

interface IncidentListItem extends GetDataPanelIncidents_currentUser_site_incidents_edges_node {}

interface IncidentsByAssigneeDataPanelProps {
  assigneeId: string;
  assigneeName: string;
  open: boolean;
  startDate: DateTime;
  endDate: DateTime;
  timezone: string;
  onClose: () => void;
}
export function IncidentsByAssigneeDataPanel({
  assigneeId,
  assigneeName,
  open,
  startDate,
  endDate,
  timezone,
  onClose,
}: IncidentsByAssigneeDataPanelProps) {
  const [dateRangeFilter, setDateRangeFilter] = useState<DateRange>({
    startDate: startDate,
    endDate: endDate,
  });
  const [fetchingMore, setFetchingMore] = useState(false);

  const filterBag: FilterBag = {
    ASSIGNMENT: { value: [assigneeId] },
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
  }, [open, getDataPanelIncidents, assigneeId]);

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
  };

  const handleClose = () => {
    setDateRangeFilter({ startDate, endDate });
    onClose();
  };

  return (
    <DataPanel open={open} onClose={handleClose}>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <Box sx={{ display: "flex", alignItems: "center" }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h2">Assigned to {assigneeName}</Typography>
          </Box>
          <Box>
            <DateRangePicker
              uiKey="button-filter-by-date-range"
              onChange={handleDateRangeChange}
              values={dateRangeFilter}
              icon={<CalendarToday sx={{ height: 16, width: 16 }} />}
              timezone={timezone}
            />
          </Box>
        </Box>
        <DataPanelIncidentList
          loading={loading}
          incidents={incidents}
          title="Most Recent Incidents"
          showLoadMore={showLoadMore}
          handleLoadMore={handleLoadMore}
          fetchingMore={fetchingMore}
          uiKey="incidents-by-assignee-data-panel"
        />
      </Box>
    </DataPanel>
  );
}
