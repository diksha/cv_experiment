import { DateTime } from "luxon";
import { BarChart as ReBarChart, YAxis, XAxis, Tooltip, Bar, ResponsiveContainer } from "recharts";
import { Box, Typography, useTheme } from "@mui/material";
import { IncidentAggregateGroupByDate, IncidentTrend, PercentageChange } from "features/dashboard";
import { TimeBucketWidth } from "__generated__/globalTypes";
import { DateRange } from "ui";
interface PayloadWrapper {
  payload: IncidentAggregateGroupByDate;
}

interface CustomTooltipProps {
  active: boolean;
  payload: PayloadWrapper[];
}

interface IncidentTrendBarChartProps {
  data: IncidentTrend;
  timezone: string;
  groupBy: TimeBucketWidth;
  title?: string;
  secondaryTitle?: string;
  showPercentage?: boolean;
  clickable?: boolean;
  onFilter1Day?: (daterange: DateRange) => void;
}

export function IncidentTrendBarChart({
  data,
  timezone,
  groupBy,
  title,
  secondaryTitle,
  showPercentage,
  clickable = true,
  onFilter1Day,
}: IncidentTrendBarChartProps) {
  const theme = useTheme();

  const CustomBarChartTooltip = (props: unknown): JSX.Element => {
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
      const timestampString =
        groupBy === TimeBucketWidth.HOUR
          ? datetime.toFormat("LLL d, h:mma, ZZZZ")
          : datetime.toFormat("ccc, LLL d, ZZZZ");
      const numIncidents = payload[0].payload.dateCount;
      return (
        <Box sx={{ border: `1px solid ${theme.palette.grey[300]}`, backgroundColor: "white", padding: "8px" }}>
          <Typography sx={{ fontWeight: "bold" }}>{timestampString}</Typography>
          <Typography>{numIncidents} incidents</Typography>
        </Box>
      );
    }
    return <></>;
  };

  const formatXAxisTick = (datetimeISO: string): string => {
    const datetime = DateTime.fromISO(datetimeISO).setZone(timezone);
    if (groupBy === TimeBucketWidth.HOUR) {
      return datetime.toFormat("h:mma");
    }
    return datetime.toFormat("LLL d");
  };

  const handleBarClick = (data: IncidentAggregateGroupByDate) => {
    if (groupBy === TimeBucketWidth.DAY && onFilter1Day) {
      onFilter1Day({
        startDate: DateTime.fromISO(data.dimensions.datetime).setZone(timezone).startOf("day"),
        endDate: DateTime.fromISO(data.dimensions.datetime).setZone(timezone).endOf("day"),
      });
    }
  };

  const hasHeader = title || secondaryTitle || showPercentage;

  return (
    <>
      {hasHeader && (
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "20px" }}>
          <Box>
            {title && (
              <Typography variant="h4" sx={{ marginBottom: "4px" }}>
                {title}
              </Typography>
            )}
            {secondaryTitle && <Typography sx={{ color: theme.palette.grey[500] }}>{secondaryTitle}</Typography>}
          </Box>
          <Box>{showPercentage && <PercentageChange trend={data} />}</Box>
        </Box>
      )}
      <ResponsiveContainer width="100%" height={200}>
        <ReBarChart data={data.mergedOneDayGroups} barGap={20}>
          <XAxis dataKey="dimensions.datetime" tickFormatter={formatXAxisTick} style={{ fontSize: 14 }} />
          <YAxis allowDecimals={false} width={36} style={{ fontSize: 14 }} />
          <Tooltip content={<CustomBarChartTooltip />} />
          <Bar
            dataKey="dateCount"
            fill={theme.palette.primary[900]}
            radius={[2, 2, 0, 0]}
            minPointSize={2}
            onClick={handleBarClick}
            style={clickable ? { cursor: groupBy === TimeBucketWidth.DAY ? "pointer" : "auto" } : {}}
          />
        </ReBarChart>
      </ResponsiveContainer>
    </>
  );
}
