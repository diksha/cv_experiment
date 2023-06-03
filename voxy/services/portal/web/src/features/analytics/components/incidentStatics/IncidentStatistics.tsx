import { DateTime } from "luxon";
import { IncidentType } from "features/analytics";
import { filteredIncidentsLink, IncidentStatisticsBarData, IncidentStatisticsBarCard } from "features/incidents";
import { useMemo } from "react";

import { GetAnalyticsPageData_analytics_series } from "__generated__/GetAnalyticsPageData";

interface SeriesDataPoint extends GetAnalyticsPageData_analytics_series {}
interface IncidentStatisticsProps {
  startDate: DateTime;
  endDate: DateTime;
  incidentTypes: IncidentType[];
  loading: boolean;
  series: SeriesDataPoint[];
}

function getIncidentTypeCounts(incidentTypes: IncidentType[], series: SeriesDataPoint[]): { [key: string]: number } {
  // Set default value from list of incident types
  const incidentTypeCounts: Record<string, number> = {};
  for (const incidentType of incidentTypes) {
    if (incidentType?.key) {
      incidentTypeCounts[incidentType.key] = 0;
    }
  }

  // Add value from series
  for (const item of series) {
    for (const key in item.incidentTypeCounts) {
      incidentTypeCounts[key] = (incidentTypeCounts[key] || 0) + item.incidentTypeCounts[key];
    }
  }
  return incidentTypeCounts;
}
export function IncidentStatistics({ startDate, endDate, incidentTypes, loading, series }: IncidentStatisticsProps) {
  const highPriorityCount = series.reduce(
    (sum: number, point: SeriesDataPoint) => sum + point.priorityCounts.highPriorityCount,
    0
  );
  const mediumPriorityCount = series.reduce(
    (sum: number, point: SeriesDataPoint) => sum + point.priorityCounts.mediumPriorityCount,
    0
  );
  const lowPriorityCount = series.reduce(
    (sum: number, point: SeriesDataPoint) => sum + point.priorityCounts.lowPriorityCount,
    0
  );

  const incidentTypeMap = useMemo(() => {
    let result: Record<string, IncidentType> = {};
    if (incidentTypes) {
      for (const incidentType of incidentTypes) {
        result[incidentType.key] = incidentType;
      }
    }
    return result;
  }, [incidentTypes]);
  const incidentTypeCounts = getIncidentTypeCounts(incidentTypes, series);
  const incidentTypeData: IncidentStatisticsBarData[] = useMemo(() => {
    const maxCount = Math.max(...Object.values(incidentTypeCounts));
    const items = Object.keys(incidentTypeCounts).map((key: string) => ({
      name: incidentTypeMap[key]?.name,
      key: key,
      count: incidentTypeCounts[key],
      maxCount: maxCount,
      barColor: incidentTypeMap[key]?.backgroundColor,
      barClassName: undefined,
      linkTo: filteredIncidentsLink({
        startDate: { value: startDate.toISO() },
        endDate: { value: endDate.toISO() },
        INCIDENT_TYPE: { value: [key] },
      }),
    }));
    return items;
  }, [incidentTypeCounts, incidentTypeMap, startDate, endDate]);

  const incidentPriorityData: IncidentStatisticsBarData[] = useMemo(() => {
    const maxCount = Math.max(highPriorityCount, mediumPriorityCount, lowPriorityCount);
    const items = [
      {
        name: "High",
        key: "high",
        count: highPriorityCount,
        maxCount: maxCount,
        barColor: "#C44E3C",
        linkTo: filteredIncidentsLink({
          startDate: { value: startDate.toISO() },
          endDate: { value: endDate.toISO() },
          PRIORITY: { value: ["high"] },
        }),
      },
      {
        name: "Medium",
        key: "medium",
        count: mediumPriorityCount,
        maxCount: maxCount,
        barColor: "#E98D69",
        linkTo: filteredIncidentsLink({
          startDate: { value: startDate.toISO() },
          endDate: { value: endDate.toISO() },
          PRIORITY: { value: ["medium"] },
        }),
      },
      {
        name: "Low",
        key: "low",
        count: lowPriorityCount,
        maxCount: maxCount,
        barColor: "#F2D51A",
        linkTo: filteredIncidentsLink({
          startDate: { value: startDate.toISO() },
          endDate: { value: endDate.toISO() },
          PRIORITY: { value: ["low"] },
        }),
      },
    ];
    return items;
  }, [highPriorityCount, mediumPriorityCount, lowPriorityCount, startDate, endDate]);

  return (
    <div className="flex gap-8 pb-8 flex-col md:flex-row">
      <IncidentStatisticsBarCard
        title="Incidents by Type"
        loading={loading}
        data={incidentTypeData}
        className="flex-1"
      />
      <IncidentStatisticsBarCard
        title="Incidents by Priority"
        loading={loading}
        data={incidentPriorityData}
        className="flex-1"
      />
    </div>
  );
}
