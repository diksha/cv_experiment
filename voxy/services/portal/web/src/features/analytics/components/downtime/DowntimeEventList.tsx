import { DataPanelSection } from "ui";
import { useEffect, useState } from "react";
import { getNodes } from "graphql/utils";
import { useLazyQuery } from "@apollo/client";
import { DateTime } from "luxon";
import { filterNullValues } from "shared/utilities/types";
import { Skeleton, Typography, Box, useTheme, FormControl, MenuItem, Select, SelectChangeEvent } from "@mui/material";
import { IncidentDetailsDataPanel, IncidentRow } from "features/incidents";
import { LoadingButton } from "@mui/lab";
import {
  GetProductionLineEvents,
  GetProductionLineEventsVariables,
  GetProductionLineEvents_productionLineDetails_incidents_edges_node,
} from "__generated__/GetProductionLineEvents";
import { GET_PRODUCTION_LINE_EVENTS } from "./queries";

interface IncidentListItem extends GetProductionLineEvents_productionLineDetails_incidents_edges_node {}

interface DowntimeEventListProps {
  productionLineId: string;
  startTimestamp: DateTime;
  endTimestamp: DateTime;
  timezone: string;
}

const DURATION_DESC = "-time_duration";
const TIMESTAMP_DESC = "-timestamp";

export function DowntimeEventList({
  productionLineId,
  startTimestamp,
  endTimestamp,
  timezone,
}: DowntimeEventListProps) {
  const theme = useTheme();
  const [fetchingMore, setFetchingMore] = useState(false);
  const [incidents, setIncidents] = useState<IncidentListItem[]>([]);
  const [orderBy, setOrderBy] = useState<string>(TIMESTAMP_DESC);
  const [getProductionLineEvents, { data, loading, fetchMore }] = useLazyQuery<
    GetProductionLineEvents,
    GetProductionLineEventsVariables
  >(GET_PRODUCTION_LINE_EVENTS, {
    fetchPolicy: "network-only",
    variables: {
      first: 5,
      productionLineId,
      startTimestamp,
      endTimestamp,
      orderBy,
    },
  });
  useEffect(() => {
    const incidentNodes = getNodes<IncidentListItem>(data?.productionLineDetails?.incidents);
    const incidents = filterNullValues<IncidentListItem>(incidentNodes);
    setIncidents(incidents);
  }, [data, data?.productionLineDetails]);

  useEffect(() => {
    getProductionLineEvents();
  }, [orderBy, getProductionLineEvents]);

  const empty = incidents.length === 0;
  const title = generateTitle(startTimestamp, endTimestamp, timezone);
  const hasNextPage = data?.productionLineDetails?.incidents?.pageInfo?.hasNextPage;
  const endCursor = hasNextPage && data?.productionLineDetails?.incidents?.pageInfo?.endCursor;
  const showLoadMore = hasNextPage && endCursor;

  const onClick = (event: SelectChangeEvent<string>) => {
    setOrderBy(event.target.value);
  };

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

  if (loading) {
    return <EventListSkeleton />;
  }

  return (
    <DataPanelSection
      title={title}
      bodyPadding={0}
      sortButton={empty ? null : <SortDropdown value={orderBy} onClick={onClick} />}
    >
      {empty ? (
        <Box>
          <Typography textAlign="center" color={theme.palette.grey[500]} padding={4}>
            No events detected during this time
          </Typography>
        </Box>
      ) : (
        <>
          {incidents.map((incident) => (
            <DowntimeEventListItem key={incident.id} incident={incident} />
          ))}
          {showLoadMore ? (
            <Box sx={{ borderTop: "1px solid", borderColor: theme.palette.grey[300] }}>
              <LoadingButton onClick={handleLoadMore} sx={{ width: "100%" }} loading={fetchingMore}>
                Load More
              </LoadingButton>
            </Box>
          ) : null}
        </>
      )}
    </DataPanelSection>
  );
}

interface DowntimeEventListItemProps {
  incident: IncidentListItem;
}
function DowntimeEventListItem({ incident }: DowntimeEventListItemProps) {
  const [dataPanelOpen, setDataPanelOpen] = useState(false);

  const handleClick = () => {
    setDataPanelOpen(true);
  };
  const handleClose = () => {
    setDataPanelOpen(false);
  };

  return (
    <>
      <IncidentRow incident={incident} active={false} onClick={handleClick} showDuration mobileOnly />
      <IncidentDetailsDataPanel incidentUuid={incident.uuid} open={dataPanelOpen} onClose={handleClose} />
    </>
  );
}

function generateTitle(startTimestamp: DateTime, endTimestamp: DateTime, timezone: string): string {
  const localizedStart = startTimestamp.setZone(timezone);
  const localizedEnd = endTimestamp.setZone(timezone);
  const sameDate = localizedStart.startOf("day").equals(localizedEnd.startOf("day"));
  const timezoneString = localizedStart.zoneName;

  if (sameDate) {
    const dateString = localizedStart.toFormat("MMM d");
    const startTimeString = localizedStart.toFormat("hh:mma");
    const endTimeString = localizedEnd.toFormat("hh:mma");
    return `${dateString}, ${startTimeString} - ${endTimeString} ${timezoneString}`;
  }

  const startString = localizedStart.toFormat("MMM d, hh:mma");
  const endString = localizedEnd.toFormat("MMM d, hh:mma");
  return `${startString} - ${endString} ${timezoneString}`;
}

function EventListSkeleton() {
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

function SortDropdown(props: { value: string; onClick: (event: SelectChangeEvent<string>) => void }) {
  const { value, onClick } = props;

  return (
    <FormControl size="small" sx={{ width: "200px" }}>
      <Select id="event-list-sort-select" autoWidth value={value} onChange={onClick}>
        <MenuItem value={TIMESTAMP_DESC}>Recent First</MenuItem>
        <MenuItem value={DURATION_DESC}>Longest First</MenuItem>
      </Select>
    </FormControl>
  );
}
