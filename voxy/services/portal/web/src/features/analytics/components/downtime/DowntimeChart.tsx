import { DateTime } from "luxon";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Brush } from "recharts";
import { ProductionLineStatusDataPoint, ChartBrushRange } from "./types";
import { toHumanRead } from "features/analytics/helpers";
import { useTheme } from "@mui/material";

export interface DowntimeChartProps {
  data: ProductionLineStatusDataPoint[];
  showBrush?: boolean;
  brushRange?: ChartBrushRange;
  timezone: string;
  greyscale?: boolean;
  onBrushChange?: (newBrushRange: ChartBrushRange) => void;
  onBarClick?: (index: number) => void;
}

interface PayloadWrapper {
  payload: ProductionLineStatusDataPoint;
  value: number;
  dataKey: string;
  color: string;
}

interface CustomTooltipProps {
  active: boolean;
  payload: PayloadWrapper[];
}

interface TooltipEntryProps {
  payload: PayloadWrapper;
  greyscale?: boolean;
}

interface ChartClickEvent {
  activeTooltipIndex?: number;
}

function TooltipEntry({ payload, greyscale }: TooltipEntryProps): JSX.Element {
  let label = payload.dataKey;
  let color = payload.color;
  if (label.includes("downtime")) {
    label = "Downtime";
    color = greyscale ? "#484848" : color;
  } else if (label.includes("uptime")) {
    label = "Uptime";
  } else if (label.includes("unknown")) {
    label = "Unknown";
    color = greyscale ? "#C9CDCF" : "#484848";
  }

  const value = toHumanRead(payload.value) || "0 sec";
  return (
    <div style={{ color }}>
      {label}: {value}
    </div>
  );
}

function formatYAxisTick(durationSeconds: number): string {
  return toHumanRead(durationSeconds).split(",").slice(0, 1).join("");
}

export function DowntimeChart({
  data,
  showBrush,
  brushRange,
  timezone,
  greyscale,
  onBrushChange,
  onBarClick,
}: DowntimeChartProps) {
  const theme = useTheme();

  const CustomTooltip = (props: unknown): JSX.Element => {
    // NOTE: Recharts type support is not very mature and doesn't support generics
    // for data/payloads and custom tooltips, so we type props as `unknown` and then
    // cast to our desired type. We lose some type safety here, but this should work
    // fine as long as we ensure the data type we pass to to the chart component stays
    // in sync with our custom tooltip prop type.
    //
    // TODO(PRO-1176): add runtime type checking to ensure custom tooltip props are correct
    const { active, payload } = props as CustomTooltipProps;
    if (active && payload && payload.length) {
      const datetime = DateTime.fromISO(payload[0].payload.dimensions.datetime).setZone(timezone);
      const timestampString = datetime.toFormat("LLL d, h:mma, ZZZZ");
      return (
        <div className="bg-white p-3" style={{ border: "1px solid #cccccc" }}>
          <div className="pb-3">{timestampString}</div>
          {payload.map((p) => (
            <TooltipEntry key={`${p.dataKey}-${p.payload.dimensions.datetime}`} payload={p} greyscale={greyscale} />
          ))}
        </div>
      );
    }
    return <></>;
  };

  const formatXAxisTick = (datetimeISO: string): string => {
    const datetime = DateTime.fromISO(datetimeISO).setZone(timezone);
    return datetime.toFormat("LLL d, ha");
  };

  const handleBrushChange = (newRange: { startIndex?: number; endIndex?: number }) => {
    if (onBrushChange) {
      const dataMinIndex = 0;
      const dataMaxIndex = data.length === 0 ? 0 : data.length - 1;
      const startIndex = newRange?.startIndex || dataMinIndex;
      const endIndex = newRange?.endIndex || dataMaxIndex;
      const validRange = startIndex <= endIndex;
      if (validRange) {
        onBrushChange({ startIndex, endIndex });
      }
    }
  };

  const handleChartClick = (event: ChartClickEvent) => {
    if (onBarClick && typeof event?.activeTooltipIndex == "number") {
      onBarClick(event.activeTooltipIndex);
    }
  };

  if (data && data.length > 0) {
    return (
      <ResponsiveContainer width="100%" height={172}>
        <BarChart height={172} barGap={3} data={data} onClick={handleChartClick}>
          <XAxis dataKey="dimensions.datetime" tickFormatter={formatXAxisTick} />
          <YAxis tickFormatter={formatYAxisTick} />
          <Tooltip content={<CustomTooltip />} />
          <Bar
            stackId="a"
            barSize={7}
            dataKey="metrics.uptimeDurationSeconds"
            fill={greyscale ? theme.palette.primary[900] : "#00C853"}
          />
          <Bar
            stackId="a"
            barSize={7}
            dataKey="metrics.downtimeDurationSeconds"
            fill={greyscale ? "#C9CDCF" : "#D32F2F"}
          />
          <Bar stackId="a" barSize={7} dataKey="metrics.unknownDurationSeconds" fill="#eeeeee" />

          {showBrush ? (
            <Brush
              dataKey="dimensions.datetime"
              height={30}
              strokeOpacity={80}
              stroke="#888888"
              travellerWidth={8}
              startIndex={brushRange?.startIndex}
              endIndex={brushRange?.endIndex}
              tickFormatter={formatXAxisTick}
              onChange={handleBrushChange}
            />
          ) : null}
        </BarChart>
      </ResponsiveContainer>
    );
  } else {
    return (
      <div className="bg-gray-100 w-full h-44 rounded-lg flex items-center justify-center">
        <div>No Data Available</div>
      </div>
    );
  }
}
