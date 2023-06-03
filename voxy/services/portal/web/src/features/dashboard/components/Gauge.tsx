import { Typography, Box, useTheme } from "@mui/material";
import { YELLOW_SCORE_THRESHOLD, GREEN_SCORE_THRESHOLD } from "./constants";

type GaugeSize = "small" | "medium" | "large";

const gaugeSizeToRadiusMap: Record<GaugeSize, number> = {
  small: 20,
  medium: 30,
  large: 50,
};

const gaugeSizeToFontSizeMap: Record<GaugeSize, string> = {
  small: "16px",
  medium: "24px",
  large: "32px",
};

interface ScoreGaugeProps {
  score: number;
  size: GaugeSize;
}

export function ScoreGauge({ score, size }: ScoreGaugeProps) {
  const theme = useTheme();
  const color =
    score > GREEN_SCORE_THRESHOLD
      ? theme.palette.success.main
      : score > YELLOW_SCORE_THRESHOLD
      ? theme.palette.warning.main
      : theme.palette.error.main;
  const radius = gaugeSizeToRadiusMap[size];
  const fontSize = gaugeSizeToFontSizeMap[size];
  return <Gauge percent={score} color={color} radius={radius} fontSize={fontSize} />;
}

interface GaugeProps {
  percent: number;
  radius: number;
  fontSize: string;
  color?: string;
  backgroundColor?: string;
}

function Gauge({ percent, radius, fontSize, color, backgroundColor }: GaugeProps) {
  const theme = useTheme();
  color = color || theme.palette.primary.main;
  backgroundColor = backgroundColor || theme.palette.grey[200];

  const strokeWidth = radius * 0.2;
  const innerRadius = radius - strokeWidth / 2;
  const circumference = innerRadius * 2 * Math.PI;
  const arc = circumference * (270 / 360);
  const dashArray = `${arc} ${circumference}`;
  const transform = `rotate(135, ${radius}, ${radius})`;
  const percentNormalized = Math.min(Math.max(percent, 0), 100);
  const offset = arc - (percentNormalized / 100) * arc;

  return (
    <Box sx={{ position: "relative", display: "inline-block" }}>
      <Box
        sx={{
          height: "100%",
          width: "100%",
          position: "absolute",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Typography sx={{ color, fontSize, fontWeight: "bold" }}>{percent}</Typography>
      </Box>
      <svg height={radius * 2} width={radius * 2}>
        <circle
          cx={radius}
          cy={radius}
          fill="transparent"
          r={innerRadius}
          stroke={backgroundColor}
          strokeWidth={strokeWidth}
          strokeDasharray={dashArray}
          transform={transform}
          strokeLinecap="round"
        />
        <circle
          cx={radius}
          cy={radius}
          fill="transparent"
          r={innerRadius}
          stroke={color}
          strokeDasharray={dashArray}
          strokeWidth={strokeWidth}
          strokeDashoffset={offset}
          transform={transform}
          strokeLinecap="round"
          style={{
            transition: "stroke-dasharray 0.3s",
          }}
        />
      </svg>
    </Box>
  );
}
