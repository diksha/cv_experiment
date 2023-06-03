import { Box, Paper, Typography, useTheme } from "@mui/material";
import {
  BAR_BACKGROUND_COLOR_GREY,
  defaultScoreTierConfig,
  EventScoreBar,
  ScoreTierConfig,
  IncidentType,
} from "features/dashboard";
import { DateTime } from "luxon";
import { useState } from "react";
import { VoxelScore } from "shared/types";
import { EventTypeSummaryDataPanel } from "../dataPanels";
import { VoxelScoreCardSkeleton } from "ui";

interface ScoresByTypeProps {
  scores: VoxelScore[];
  incidentTypes: IncidentType[];
  startDate: DateTime;
  endDate: DateTime;
  loading: boolean;
  config?: ScoreTierConfig;
}
export function ScoresByType({ scores, incidentTypes, startDate, endDate, loading, config }: ScoresByTypeProps) {
  const theme = useTheme();
  const [dataPanelOpen, setDataPanelOpen] = useState(false);
  const [dataPanelCurrentEventType, setDataPanelCurrentEventType] = useState<IncidentType>();
  const scoreTierConfig = config || defaultScoreTierConfig;
  // TODO(hq): sort on BE
  const sortedScores = scores.sort((a, b) => a.value - b.value);

  const onClick = (score: VoxelScore) => {
    const type = incidentTypes.find((elem) => elem.name === score.label);
    setDataPanelOpen(true);
    setDataPanelCurrentEventType(type);
  };

  const handleDataPanelClose = () => {
    setDataPanelOpen(false);
    setDataPanelCurrentEventType(undefined);
  };

  if (loading) {
    return <VoxelScoreCardSkeleton showHeader />;
  }

  return (
    <>
      <Paper sx={{ minWidth: 300 }} data-ui-key="voxel-score-by-type-card">
        <Box sx={{ paddingY: 3, paddingX: 4 }}>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, paddingBottom: 2 }}>
            <Typography variant="h4">Voxel Score by Type</Typography>
            <Typography sx={{ color: theme.palette.grey[500] }}>Voxel scores for your organization</Typography>
          </Box>
          <Box display="flex" flexDirection="column" gap={4} paddingBottom={4}>
            {sortedScores.map((eventScore) => {
              return (
                <EventScoreBar
                  key={eventScore.label}
                  label={eventScore.label}
                  score={eventScore.value}
                  config={scoreTierConfig}
                  barBgColor={BAR_BACKGROUND_COLOR_GREY}
                  onClick={onClick}
                />
              );
            })}
            {!sortedScores.length && (
              <Box
                sx={{
                  width: "100%",
                  textAlign: "center",
                  paddingTop: 3,
                  paddingX: 3,
                  color: theme.palette.grey[500],
                }}
              >
                No Voxel Scores during this time
              </Box>
            )}
          </Box>
        </Box>
      </Paper>
      {dataPanelCurrentEventType && (
        <EventTypeSummaryDataPanel
          open={dataPanelOpen}
          startDate={startDate}
          endDate={endDate}
          eventType={dataPanelCurrentEventType}
          onClose={handleDataPanelClose}
        />
      )}
    </>
  );
}
