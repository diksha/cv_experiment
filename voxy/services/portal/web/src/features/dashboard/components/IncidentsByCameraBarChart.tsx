import React from "react";
import { DateTime } from "luxon";
import { BarChart as ReBarChart, YAxis, XAxis, Tooltip, Bar, ResponsiveContainer } from "recharts";
import { Box, Chip, Typography, useTheme } from "@mui/material";
import { TimeBucketWidth } from "__generated__/globalTypes";
import { TotalsByDate, IncidentType, INCIDENT_TYPE_TEXT_COLORS } from "features/dashboard";
import { CheckCircle, Circle } from "@mui/icons-material";
import { DateRange } from "ui";

interface PayloadWrapper {
  payload: TotalsByDate;
}

interface CustomTooltipProps {
  active: boolean;
  payload: PayloadWrapper[];
}

interface IncidentsByCameraBarChartProps {
  data: TotalsByDate[];
  timezone: string;
  groupBy: TimeBucketWidth;
  incidentTypes: IncidentType[];
  filteredIncidentTypes: string[];
  setFilteredIncidentTypes: React.Dispatch<React.SetStateAction<string[]>>;
  onFilter1Day: (daterange: DateRange) => void;
}

export function IncidentsByCameraBarChart({
  data,
  timezone,
  groupBy,
  incidentTypes,
  filteredIncidentTypes,
  setFilteredIncidentTypes,
  onFilter1Day,
}: IncidentsByCameraBarChartProps) {
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
      const datetime = DateTime.fromISO(payload[0].payload.datetime).setZone(timezone);
      const timestampString =
        groupBy === TimeBucketWidth.HOUR
          ? datetime.toFormat("LLL d, h:mma, ZZZZ")
          : datetime.toFormat("ccc, LLL d, ZZZZ");
      const sum = Object.values(payload[0].payload.totals).reduce((a, b) => a + b, 0);

      return (
        <Box p={1} sx={{ border: `1px solid ${theme.palette.grey[300]}`, backgroundColor: "white" }}>
          <Typography sx={{ fontWeight: 700 }}>{timestampString}</Typography>
          <Box sx={{ display: "flex" }}>
            <Typography sx={{ fontWeight: 700, minWidth: "12px", textAlign: "center" }}>{sum}</Typography>
            <Typography>&nbsp;Total incidents</Typography>
          </Box>
          {incidentTypes.map((incident) => {
            if (filteredIncidentTypes.includes(incident.key)) {
              return (
                <Box key={incident.key} sx={{ display: "flex" }}>
                  <Typography
                    sx={{ color: incident.backgroundColor, fontWeight: 700, minWidth: "12px", textAlign: "center" }}
                  >
                    {payload[0].payload.totals[incident.key]}
                  </Typography>
                  <Typography>&nbsp;{incident.name}</Typography>
                </Box>
              );
            }
            return <React.Fragment key={incident.key}></React.Fragment>;
          })}
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

  const handleBarClick = (data: TotalsByDate) => {
    if (groupBy === TimeBucketWidth.DAY) {
      onFilter1Day({
        startDate: DateTime.fromISO(data.datetime).setZone(timezone).startOf("day"),
        endDate: DateTime.fromISO(data.datetime).setZone(timezone).endOf("day"),
      });
    }
  };

  const onClickChip = (incident: IncidentType) => {
    if (filteredIncidentTypes.includes(incident.key)) {
      setFilteredIncidentTypes(filteredIncidentTypes.filter((i) => i !== incident.key));
    } else {
      setFilteredIncidentTypes([...filteredIncidentTypes, incident.key]);
    }
  };

  return (
    <>
      <ResponsiveContainer width="100%" height={200}>
        <ReBarChart data={data} barGap={20}>
          <XAxis dataKey="datetime" tickFormatter={formatXAxisTick} style={{ fontSize: 14 }} />
          <YAxis allowDecimals={false} width={32} style={{ fontSize: 14 }} />
          <Tooltip wrapperStyle={{ zIndex: 10 }} content={<CustomBarChartTooltip />} />
          {incidentTypes.map((incident) => {
            return (
              <Bar
                key={incident.key}
                stackId="a"
                dataKey={`totals.${incident.key}`}
                fill={incident.backgroundColor || theme.palette.common.black}
                onClick={handleBarClick}
                style={{ cursor: groupBy === TimeBucketWidth.DAY ? "pointer" : "auto" }}
              />
            );
          })}
        </ReBarChart>
      </ResponsiveContainer>
      {incidentTypes.map((incident) => {
        const active = filteredIncidentTypes.includes(incident.key);
        return (
          <Chip
            key={incident.key}
            icon={
              active ? (
                <CheckCircle style={{ color: theme.palette.common.white }} />
              ) : (
                <Circle style={{ color: theme.palette.common.white }} />
              )
            }
            label={incident.name}
            sx={{
              backgroundColor: active ? incident.backgroundColor : "#EDF0F4",
              color: active ? INCIDENT_TYPE_TEXT_COLORS[incident.key] : theme.palette.common.black,
              marginRight: "8px",
              marginBottom: "4px",
              cursor: "pointer",
              "&:hover": { backgroundColor: active ? incident.backgroundColor : "#EDF0F4" },
            }}
            onClick={() => onClickChip(incident)}
          />
        );
      })}
    </>
  );
}
