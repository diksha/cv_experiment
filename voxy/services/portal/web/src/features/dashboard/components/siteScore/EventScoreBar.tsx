import { styled } from "@mui/material/styles";
import Box from "@mui/material/Box";
import { BAR_BACKGROUND_COLOR } from "./constants";
import { LinearProgress, linearProgressClasses } from "@mui/material";
import { ScoreTier, EventScore, ScoreTierConfig } from "./types";
import { getScoreTier } from "./helpers";
import { VoxelScore } from "shared/types";

const backgroundBarSelector = `&.${linearProgressClasses.colorPrimary}`;
const foregroundBarSelector = `& .${linearProgressClasses.bar}`;

const generateColoredBar = (tier: ScoreTier, barBgColor: string) =>
  styled(LinearProgress)(() => ({
    height: 6,
    borderRadius: 3,
    [backgroundBarSelector]: {
      backgroundColor: barBgColor,
    },
    [foregroundBarSelector]: {
      borderRadius: 3,
      backgroundColor: tier.color,
    },
  }));

interface EventScoreBarProps extends EventScore {
  config: ScoreTierConfig;
  barBgColor?: string;
  onClick?: (score: VoxelScore) => void;
}
export function EventScoreBar({
  label,
  score,
  config,
  barBgColor = BAR_BACKGROUND_COLOR,
  onClick,
}: EventScoreBarProps) {
  const tier = getScoreTier(score, config);
  const Bar = generateColoredBar(tier, barBgColor);

  return (
    <Box
      onClick={onClick ? () => onClick({ label, value: score }) : undefined}
      sx={{ cursor: onClick ? "pointer" : "auto" }}
      data-ui-key="voxel-score-bar"
    >
      <Box display="flex" gap={2} fontWeight="bold">
        <Box flex="1">{label}</Box>
        <Box>{score}</Box>
      </Box>
      <Bar variant="determinate" value={score} />
    </Box>
  );
}
