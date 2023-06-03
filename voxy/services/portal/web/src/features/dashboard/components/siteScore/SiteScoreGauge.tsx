import { Box, Typography, useTheme } from "@mui/material";
import { ScoreTierConfig } from "./types";
import { getScoreTier } from "./helpers";
import { ReactNode } from "react";
import { HelpRounded } from "@mui/icons-material";
import { StyledTooltip } from "ui";

const SVG_VIEWBOX_HEIGHT = 60;
const SVG_VIEWBOX_WIDTH = 100;
const RADIUS = 40;

interface GaugeProps {
  score: number;
  config: ScoreTierConfig;
  title?: string;
  mode?: "dark" | "light";
}
export function SiteScoreGauge({ score, config, title = "Voxel Score", mode = "dark" }: GaugeProps) {
  const theme = useTheme();
  // TODO: validate score and generate arc size based on config
  const sanitizedScore = Math.min(100, Math.max(0, score));
  const tier = getScoreTier(sanitizedScore, config);

  return (
    <>
      <svg
        style={{ width: "100%", maxWidth: "250px", margin: "0 auto" }}
        viewBox={`0 0 ${SVG_VIEWBOX_WIDTH} ${SVG_VIEWBOX_HEIGHT}`}
        preserveAspectRatio="xMidYMid meet"
      >
        <Arc
          arcDegrees={180}
          arcDegreesOffset={0}
          percent={sanitizedScore}
          showDot={true}
          color={tier.color}
          mode={mode}
        />
        <TextElem text={sanitizedScore} mode={mode} x="50%" y="75%" fontSize={20} />
        <TextElem text={0} mode={mode} x="12%" y="94%" fontSize={5} />
        <TextElem text={100} mode={mode} x="88%" y="94%" fontSize={5} />
      </svg>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          paddingLeft: "18px",
          position: "relative",
          top: "-8px",
          marginBottom: "-8px",
        }}
      >
        <Typography
          sx={{
            color: mode === "dark" ? theme.palette.grey[100] : theme.palette.primary.main,
            fontWeight: "bold",
            fontSize: "16px",
            marginRight: "4px",
          }}
        >
          {title}
        </Typography>
        <StyledTooltip
          title="A higher score is better. A higher score indicates less observed risk by Voxel."
          placement="top"
        >
          <HelpRounded sx={{ width: 14, height: 14, opacity: 0.25 }} />
        </StyledTooltip>
      </Box>
    </>
  );
}

interface ArcProps {
  percent: number;
  color: string;
  arcDegrees: number;
  arcDegreesOffset: number;
  showDot: boolean;
  mode?: "dark" | "light";
}
function Arc({ percent, color, arcDegrees, arcDegreesOffset, showDot, mode = "dark" }: ArcProps) {
  const theme = useTheme();
  const backgroundColor = mode === "dark" ? theme.palette.grey[200] : theme.palette.grey[500];

  const percentNormalized = Math.min(Math.max(percent, 0), 100);
  const strokeWidth = RADIUS * 0.1;
  const innerRadius = RADIUS - strokeWidth / 2;
  const circumference = innerRadius * 2 * Math.PI;

  const centerX = SVG_VIEWBOX_WIDTH / 2;
  const centerY = SVG_VIEWBOX_HEIGHT * 0.8;

  const arc = circumference * (arcDegrees / 360);
  const dashArray = `${arc} ${circumference}`;
  const offset = arc - (percentNormalized / 100) * arc;

  // Rotate this arc based on the provided offset
  const arcRotationDegrees = 180 + arcDegreesOffset;
  const transform = `rotate(${arcRotationDegrees}, ${centerX}, ${centerY})`;

  // Find coordinates of indicator dot
  const percentFillDegrees = arcDegrees * (percentNormalized / 100);

  // Determine how many degrees of buffer we need to ensure the dot stroke lines up with the arc edge
  // Without a buffer, the dot overlaps too much with gaps and we lose the visual gap between arcs
  const dotRadius = strokeWidth / 2;
  const dotStrokeWidth = dotRadius / 1.25;
  const strokeWidthDegrees = dotRadius / (circumference / 360);
  const originalDotRotationDegrees = arcRotationDegrees + percentFillDegrees;
  const minDotRotationDegrees = arcRotationDegrees + strokeWidthDegrees;
  const maxDotRotationDegrees = arcRotationDegrees + arcDegrees - strokeWidthDegrees;
  const safeDotRotationDegrees = Math.min(
    Math.max(originalDotRotationDegrees, minDotRotationDegrees),
    maxDotRotationDegrees
  );

  // Calculate X/Y coordinates of indicator dot
  const dotX = centerX + innerRadius * Math.cos((safeDotRotationDegrees * Math.PI) / 180);
  const dotY = centerY + innerRadius * Math.sin((safeDotRotationDegrees * Math.PI) / 180);

  return (
    <>
      <circle
        cx={centerX}
        cy={centerY}
        fill="transparent"
        r={innerRadius}
        stroke={backgroundColor}
        strokeOpacity={0.2}
        strokeWidth={strokeWidth}
        strokeDasharray={dashArray}
        transform={transform}
        strokeLinecap="round"
      />
      <circle
        cx={centerX}
        cy={centerY}
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
      {showDot ? (
        <circle
          cx={dotX}
          cy={dotY}
          fill={theme.palette.grey[100]}
          r={strokeWidth}
          stroke={color}
          strokeWidth={dotStrokeWidth}
        />
      ) : null}
    </>
  );
}

interface TextElemProps {
  text: string | number;
  mode?: "dark" | "light";
  x: string;
  y: string;
  fontSize: number;
  children?: ReactNode;
}
function TextElem({ text, mode = "dark", x, y, fontSize, children }: TextElemProps) {
  const theme = useTheme();

  return (
    <text
      x={x}
      y={y}
      textAnchor="middle"
      style={{
        fontSize,
        fill: mode === "dark" ? theme.palette.grey[100] : theme.palette.primary.main,
        fontWeight: "bold",
      }}
    >
      {children || text}
    </text>
  );
}
