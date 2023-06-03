import { Box, Button, Typography, Paper } from "@mui/material";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@apollo/client";
import { GetCurrentZoneIncidentFeed } from "__generated__/GetCurrentZoneIncidentFeed";
import { FeedItemData, GET_CURRENT_ZONE_INCIDENT_FEED } from "features/incidents";
import { DateTime } from "luxon";
import { useCallback, useMemo } from "react";
import { getNodes } from "graphql/utils";
import { HighlightedVideoList } from "./HighlightedVideoList";

interface HighlightedEventsProps {
  viewAllButtonPosition?: "top" | "bottom";
}
export function HighlightedEvents({ viewAllButtonPosition = "top" }: HighlightedEventsProps) {
  const navigate = useNavigate();
  // Note: This is a temp hacky solution using the GET_CURRENT_ZONE_INCIDENT_FEED as a temp quey solution.
  // Future improvement could be
  // 1. Move the query to parent level
  // 2. Make a new generic incident graphql schema
  const handleOnClick = useCallback(() => {
    navigate(`/incidents?EXTRAS=%5B"HIGHLIGHTED"%5D`);
  }, [navigate]);
  const { data, loading } = useQuery<GetCurrentZoneIncidentFeed>(GET_CURRENT_ZONE_INCIDENT_FEED, {
    fetchPolicy: "network-only",
    nextFetchPolicy: "cache-first",
    variables: {
      startDate: DateTime.now().minus({ days: 30 }).toISODate(),
      endDate: DateTime.now().toISODate(),
      filters: [{ key: "EXTRAS", valueJson: JSON.stringify(["HIGHLIGHTED"]) }],
      timeBucketSizeHours: 24,
    },
  });

  const feedItems: FeedItemData[] = useMemo(() => {
    return getNodes<FeedItemData>(data?.currentUser?.zone?.incidentFeed) || [];
  }, [data]);
  const empty = !loading && feedItems.length === 0;

  if (loading || empty) {
    return null;
  }

  return (
    <Paper sx={{ padding: 3 }} data-ui-key="highlighted-events">
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <Box sx={{ flexGrow: 1 }}>
          <Typography variant="h4">Highlighted Incidents</Typography>
        </Box>
        {viewAllButtonPosition === "top" ? (
          <Box sx={{ flexGrow: 0 }}>
            <Button size="small" onClick={handleOnClick} data-ui-key="highlighted-events-view-all">
              View All
            </Button>
          </Box>
        ) : null}
      </Box>
      <Box
        sx={{ display: "flex", justifyContent: "flex-start", paddingTop: 1, flexWrap: { xs: "wrap", md: "nowrap" } }}
      >
        <HighlightedVideoList items={feedItems} />
        {viewAllButtonPosition === "bottom" ? (
          <Button
            sx={{ width: "100%", marginTop: 2 }}
            onClick={handleOnClick}
            data-ui-key="highlighted-events-view-all"
          >
            View All
          </Button>
        ) : null}
      </Box>
    </Paper>
  );
}
