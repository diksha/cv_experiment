import React, { useMemo } from "react";
import { DateTime } from "luxon";
import { BarChart as ReBarChart, YAxis, XAxis, Tooltip, Bar, ResponsiveContainer } from "recharts";
import { Box, Chip, Typography, useTheme } from "@mui/material";
import { TimeBucketWidth } from "__generated__/globalTypes";
import { TotalsByDate } from "./types";
import { Camera, CAMERA_COLORS } from "features/dashboard";
import { CheckCircle, Circle } from "@mui/icons-material";
import { ColorConfig } from "shared/types";
import { DateRange } from "ui";

interface PayloadWrapper {
  payload: TotalsByDate;
}

interface CustomTooltipProps {
  active: boolean;
  payload: PayloadWrapper[];
}

interface IncidentTrendByCameraBarChartProps {
  data: TotalsByDate[];
  timezone: string;
  groupBy: TimeBucketWidth;
  cameras: Camera[];
  filteredCameraIds: string[];
  setFilteredCameras: React.Dispatch<React.SetStateAction<string[]>>;
  onFilter1Day: (daterange: DateRange) => void;
}

export function IncidentTrendByCameraBarChart({
  data,
  timezone,
  groupBy,
  cameras,
  filteredCameraIds,
  setFilteredCameras,
  onFilter1Day,
}: IncidentTrendByCameraBarChartProps) {
  const theme = useTheme();

  const cameraColors = useMemo(() => {
    const result: Record<string, ColorConfig> = {};
    for (let i = 0; i < cameras.length; i++) {
      result[cameras[i].id] = CAMERA_COLORS[i % CAMERA_COLORS.length];
    }
    return result;
  }, [cameras]);

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
          {cameras.map((camera) => {
            if (filteredCameraIds.includes(camera.id)) {
              const total = payload[0].payload.totals[camera.id];
              if (total) {
                return (
                  <Box key={camera.id} sx={{ display: "flex" }}>
                    <Typography
                      sx={{
                        color: cameraColors[camera.id].fill,
                        fontWeight: 700,
                        minWidth: "12px",
                        textAlign: "center",
                      }}
                    >
                      {total}
                    </Typography>
                    <Typography>&nbsp;{camera.name}</Typography>
                  </Box>
                );
              }
            }
            return <React.Fragment key={camera.id}></React.Fragment>;
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

  const onClickChip = (camera: Camera) => {
    if (filteredCameraIds.includes(camera.id)) {
      setFilteredCameras(filteredCameraIds.filter((c) => c !== camera.id));
    } else {
      setFilteredCameras([...filteredCameraIds, camera.id]);
    }
  };

  return (
    <>
      <ResponsiveContainer width="100%" height={200}>
        <ReBarChart data={data} barGap={20}>
          <XAxis dataKey="datetime" tickFormatter={formatXAxisTick} style={{ fontSize: 14 }} />
          <YAxis allowDecimals={false} width={32} style={{ fontSize: 14 }} />
          <Tooltip wrapperStyle={{ zIndex: 10 }} content={<CustomBarChartTooltip />} />
          {cameras.map((camera) => {
            return (
              <Bar
                key={camera.id}
                stackId="a"
                dataKey={`totals.${camera.id}`}
                fill={cameraColors[camera.id].fill}
                onClick={handleBarClick}
                style={{ cursor: groupBy === TimeBucketWidth.DAY ? "pointer" : "auto" }}
              />
            );
          })}
        </ReBarChart>
      </ResponsiveContainer>
      {cameras.map((camera) => {
        const active = filteredCameraIds.includes(camera.id);
        return (
          <Chip
            key={camera.id}
            icon={active ? <CheckCircle style={{ color: "#ffffff" }} /> : <Circle style={{ color: "#ffffff" }} />}
            label={camera.name}
            sx={{
              backgroundColor: active ? cameraColors[camera.id].fill : "#EDF0F4",
              color: active ? cameraColors[camera.id].text : "#000000",
              marginRight: "8px",
              marginBottom: "4px",
              cursor: "pointer",
              "&:hover": { backgroundColor: active ? cameraColors[camera.id].fill : "#EDF0F4" },
            }}
            onClick={() => onClickChip(camera)}
          />
        );
      })}
    </>
  );
}
