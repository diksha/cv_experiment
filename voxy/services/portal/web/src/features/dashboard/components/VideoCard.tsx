import { Paper, useTheme } from "@mui/material";
import { Box, Typography } from "@mui/material";
import { GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets_latestIncidents } from "__generated__/GetCurrentZoneIncidentFeed";
import { IncidentDetailsDataPanel } from "features/incidents";
import { DateTime } from "luxon";
import { useState } from "react";
import { analytics } from "shared/utilities/analytics";

interface VideoCardProps {
  incident: GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets_latestIncidents;
}

export function VideoCard({ incident }: VideoCardProps) {
  const theme = useTheme();
  const { incidentType, camera, timestamp } = incident;
  const [dataPanelOpen, setDataPanelOpen] = useState(false);
  const handleClick = () => {
    setDataPanelOpen(true);
    analytics.trackCustomEvent("datapanelHighlightedIncident");
  };
  const handleClose = () => {
    setDataPanelOpen(false);
  };

  return (
    <Box sx={{ padding: 0.5, width: "100%", minWidth: "100px", maxWidth: { md: "250px" } }}>
      <IncidentDetailsDataPanel incidentUuid={incident.uuid} open={dataPanelOpen} onClose={handleClose} />
      <Paper variant="outlined" sx={{ ":hover": { boxShadow: 1 } }}>
        <Box onClick={handleClick} sx={{ cursor: "pointer", padding: 1 }} data-ui-key="datapanel-highlighted-incident">
          <Box
            sx={{
              position: "relative",
              display: "flex",
              width: "100%",
              paddingBottom: "60%",
              backgroundColor: theme.palette.grey[800],
              overflow: "hidden",
              borderRadius: "8px",
            }}
          >
            <Box
              sx={{
                position: "absolute",
                top: 0,
                right: 0,
                bottom: 0,
                left: 0,
              }}
            >
              <img
                style={{
                  objectFit: "cover",
                  marginLeft: "auto",
                  marginRight: "auto",
                }}
                alt={`${camera?.name}-thumbnail`}
                src={camera?.thumbnailUrl || ""}
              />
            </Box>
          </Box>
          <Box sx={{ padding: "0.25rem 0rem" }}>
            <Typography sx={{ fontWeight: "bold" }}>{incidentType?.name}</Typography>
            <Typography color={theme.palette.grey[600]}>{camera?.name}</Typography>
            <Typography color={theme.palette.grey[600]}>{DateTime.fromISO(timestamp).toRelative()}</Typography>
          </Box>
        </Box>
      </Paper>
    </Box>
  );
}
