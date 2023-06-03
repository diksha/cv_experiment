import { Paper, Box, useTheme, BoxProps } from "@mui/material";
import {
  BAR_BACKGROUND_COLOR,
  BAR_BACKGROUND_COLOR_GREY,
  defaultScoreTierConfig,
  EventScoreBar,
  ScoreTierConfig,
  SiteScoreGauge,
} from "features/dashboard";
import { VoxelScore } from "shared/types";
import { VoxelScoreCardSkeleton } from "ui/skeleton";

interface VoxelScoreCardProps {
  overallScore: VoxelScore | null | undefined;
  scores: VoxelScore[];
  loading: boolean;
  title?: string;
  config?: ScoreTierConfig;
  minWidth?: number | string;
  mode?: "dark" | "light";
  boxContainerProps?: BoxProps;
  skeletonPadding?: number | string;
  onClick?: (score: VoxelScore) => void;
}
export function VoxelScoreCard({
  overallScore,
  scores,
  loading,
  config,
  title,
  minWidth = 300,
  mode = "dark",
  boxContainerProps = { padding: 2 },
  skeletonPadding,
  onClick,
}: VoxelScoreCardProps) {
  const theme = useTheme();
  const scoreTierConfig = config || defaultScoreTierConfig;
  // TODO(hq): sort on BE
  const sortedScores = scores.sort((a, b) => a.value - b.value);

  if (loading) {
    return <VoxelScoreCardSkeleton showOverallScore minWidth={minWidth} padding={skeletonPadding} />;
  }
  if (!overallScore) {
    return <></>;
  }
  return (
    <Paper
      sx={{
        backgroundColor: mode === "dark" ? theme.palette.primary.main : theme.palette.common.white,
        color: mode === "dark" ? theme.palette.grey[100] : theme.palette.primary.main,
        minWidth,
      }}
      data-ui-key="voxel-score-card"
    >
      <Box {...boxContainerProps}>
        <SiteScoreGauge score={overallScore.value} config={scoreTierConfig} title={title} mode={mode} />
        <Box display="flex" flexDirection="column" gap={4} paddingX={2} paddingY={4}>
          {sortedScores.map((eventScore) => {
            return (
              <EventScoreBar
                key={eventScore.label}
                label={eventScore.label}
                score={eventScore.value}
                config={scoreTierConfig}
                barBgColor={mode === "dark" ? BAR_BACKGROUND_COLOR : BAR_BACKGROUND_COLOR_GREY}
                {...(onClick ? { onClick } : {})}
              />
            );
          })}
        </Box>
      </Box>
    </Paper>
  );
}
