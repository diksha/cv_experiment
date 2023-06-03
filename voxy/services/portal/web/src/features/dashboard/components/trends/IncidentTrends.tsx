import { DateTime } from "luxon";
import { useMemo, useState, Fragment } from "react";
import { IncidentAggregateGroup, IncidentType, Camera, HIDDEN_TREND_INCIDENT_TYPES } from "features/dashboard";
import { IncidentTrend, IncidentTrendsMapping } from "./types";
import { IncidentTrendSkeleton } from "./IncidentTrendSkeleton";
import { IncidentTrendDataPanel } from "./IncidentTrendDataPanel";
import { Box } from "@mui/material";
import { fillIncidentAggregateGroups } from "./utils";
import { TimeBucketWidth } from "__generated__/globalTypes";
import { TrendCard } from "ui";

interface IncidentTrendsProps {
  groups: IncidentAggregateGroup[];
  incidentTypes: IncidentType[];
  loading: boolean;
  startDate: DateTime;
  endDate: DateTime;
  timezone: string;
  cameras: Camera[];
}

export function IncidentTrends({
  groups = [],
  incidentTypes,
  loading,
  startDate,
  endDate,
  timezone,
  cameras,
}: IncidentTrendsProps) {
  const trends: IncidentTrend[] = useMemo(() => {
    const mapping: IncidentTrendsMapping = fillIncidentAggregateGroups(
      groups,
      incidentTypes,
      TimeBucketWidth.DAY,
      startDate,
      endDate,
      timezone
    );
    return Object.values(mapping).sort((a, b) => b.countTotal - a.countTotal);
  }, [groups, incidentTypes, startDate, endDate, timezone]);

  return (
    <>
      {loading ? (
        <>
          <IncidentTrendSkeleton />
          <IncidentTrendSkeleton />
          <IncidentTrendSkeleton />
        </>
      ) : (
        <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {trends.map((trend) => {
            if (HIDDEN_TREND_INCIDENT_TYPES[trend.incidentTypeKey]) {
              return <Fragment key={trend.incidentTypeKey}></Fragment>;
            }
            return (
              <IncidentTrendItem
                key={trend.incidentTypeKey}
                trend={trend}
                startDate={startDate}
                endDate={endDate}
                timezone={timezone}
                cameras={cameras}
              />
            );
          })}
        </Box>
      )}
    </>
  );
}

interface IncidentTrendItemProps {
  trend: IncidentTrend;
  startDate: DateTime;
  endDate: DateTime;
  timezone: string;
  cameras: Camera[];
}

function IncidentTrendItem({ trend, startDate, endDate, timezone, cameras }: IncidentTrendItemProps) {
  const [dataPanelOpen, setDataPanelOpen] = useState(false);

  const handleClick = () => {
    setDataPanelOpen(true);
  };

  const handleDataPanelClose = () => {
    setDataPanelOpen(false);
  };

  return (
    <>
      <TrendCard
        key={trend.name}
        trend={trend}
        title={trend.name}
        secondaryTitle="Total Incidents - Last 30 Days"
        onClick={handleClick}
      />
      <IncidentTrendDataPanel
        trend={trend}
        open={dataPanelOpen}
        startDate={startDate}
        endDate={endDate}
        timezone={timezone}
        cameras={cameras}
        onClose={handleDataPanelClose}
      />
    </>
  );
}
