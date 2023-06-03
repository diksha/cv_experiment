import { Paper, Box, useTheme } from "@mui/material";
import { EventScoreBar } from "./EventScoreBar";
import { SiteScoreGauge } from "./SiteScoreGauge";
import { defaultScoreTierConfig } from "./constants";
import { filterNullValues } from "shared/utilities/types";
import { ScoreTierConfig } from "./types";
import {
  GetDashboardData_currentUser_site_overallScore,
  GetDashboardData_currentUser_site_eventScores,
} from "__generated__/GetDashboardData";

interface SiteScoreCardProps {
  overallScore: GetDashboardData_currentUser_site_overallScore;
  eventScores: (GetDashboardData_currentUser_site_eventScores | null)[];
  config?: ScoreTierConfig;
}
export function SiteScoreCard({ overallScore, eventScores, config }: SiteScoreCardProps) {
  const theme = useTheme();
  const scoreTierConfig = config || defaultScoreTierConfig;
  // TODO(hq): sort on BE
  const sorted = filterNullValues<GetDashboardData_currentUser_site_eventScores>(eventScores).sort(
    (a, b) => a.value - b.value
  );

  return eventScores.length > 0 ? (
    <Paper
      sx={{ backgroundColor: theme.palette.primary.main, color: theme.palette.grey[100], minWidth: 300 }}
      data-ui-key="site-score-card"
    >
      <Box padding={2}>
        <SiteScoreGauge score={overallScore.value} config={scoreTierConfig} />
        <Box display="flex" flexDirection="column" gap={4} paddingX={2} paddingY={4}>
          {sorted.map((eventScore) => (
            <EventScoreBar
              key={eventScore.label}
              label={eventScore.label}
              score={eventScore.value}
              config={scoreTierConfig}
            />
          ))}
        </Box>
      </Box>
    </Paper>
  ) : null;
}
