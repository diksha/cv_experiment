import {
  IncidentTrendBarChartSimple,
  IncidentType,
  PercentageChange,
  TableCard,
  TableColumnDefinition,
} from "features/dashboard";
import { useMemo, useState } from "react";
import { DateTime } from "luxon";
import { readableDaterange } from "shared/utilities/dateutil";
import { OrgSite } from "features/executive-dashboard";
import {
  fillIncidentAggregateGroupsMergeSitesIncidents,
  sortByTrendThenScore,
} from "features/dashboard/components/trends/utils";
import { Box, BoxProps, useMediaQuery, useTheme } from "@mui/material";
import { EventTypeSummaryDataPanel } from "../dataPanels";
import { TableSkeleton } from "ui";

interface RowData {
  incidentName: string;
  component: any;
}
interface TrendsByTypeProps {
  sites: OrgSite[];
  incidentTypes: IncidentType[];
  startDate: DateTime;
  endDate: DateTime;
  loading: boolean;
  showChart?: boolean;
  isoDateOnly?: boolean;
  timezone?: string;
  clickable?: boolean;
  percentageBoxProps?: BoxProps;
  tableBoxProps?: BoxProps;
  skeletonPadding?: number | string;
}
export function TrendsByType({
  sites,
  incidentTypes,
  startDate,
  endDate,
  loading,
  showChart = true,
  isoDateOnly = true,
  timezone,
  clickable,
  percentageBoxProps = { minWidth: "80px", paddingLeft: "20px", justifyContent: "space-between", alignItems: "center" },
  tableBoxProps,
  skeletonPadding,
}: TrendsByTypeProps) {
  const theme = useTheme();
  const [dataPanelOpen, setDataPanelOpen] = useState(false);
  const [dataPanelCurrentEventType, setDataPanelCurrentEventType] = useState<IncidentType>();
  const dateText = readableDaterange(startDate, endDate, "to") || "";
  const tz = timezone || DateTime.now().zoneName;
  const smBreakpoint = useMediaQuery(theme.breakpoints.up("sm"));

  const columns: TableColumnDefinition<RowData, keyof RowData>[] = [
    {
      key: "incidentName",
      header: "Type",
    },
    {
      key: "component",
      header: "Trend",
      textAlign: "right",
    },
  ];

  const rows: RowData[] = useMemo(() => {
    const mergedMap = fillIncidentAggregateGroupsMergeSitesIncidents(
      sites,
      incidentTypes,
      startDate,
      endDate,
      tz,
      isoDateOnly
    );
    const result = sortByTrendThenScore(mergedMap);
    return result.map((trend) => {
      const incidentName = incidentTypes.find((a) => a.key === trend.incidentTypeKey)?.name as string;
      const component = (
        <Box
          key={trend.incidentTypeKey}
          sx={{
            display: "flex",
            justifyContent: "space-between",
            maxWidth: smBreakpoint ? "320px" : "190px",
            margin: "0 0 0 auto",
            alignItems: "center",
            flexDirection: "row-reverse",
          }}
        >
          <PercentageChange
            trend={trend}
            boxProps={percentageBoxProps}
            textProps={{ fontSize: "14px", fontWeight: "400" }}
          />
          {showChart && <IncidentTrendBarChartSimple data={trend.mergedOneDayGroups} height={30} />}
        </Box>
      );
      return {
        incidentName,
        component,
      };
    });
  }, [incidentTypes, endDate, sites, startDate, tz, smBreakpoint, isoDateOnly, showChart, percentageBoxProps]);

  const onClick = (row: RowData) => {
    const type = incidentTypes.find((elem) => elem.name === row.incidentName);
    setDataPanelOpen(true);
    setDataPanelCurrentEventType(type);
  };

  const handleDataPanelClose = () => {
    setDataPanelOpen(false);
    setDataPanelCurrentEventType(undefined);
  };

  if (loading) {
    return <TableSkeleton padding={skeletonPadding} />;
  }

  return (
    <>
      <TableCard
        title="Trends by Type"
        subtitle={dateText}
        data={rows}
        columns={columns}
        emptyMessage="No trends during this time"
        {...(clickable ? { onRowClick: onClick } : {})}
        uiKey="trends-by-type-card"
        boxProps={tableBoxProps}
      />
      {dataPanelCurrentEventType && (
        <EventTypeSummaryDataPanel
          open={dataPanelOpen}
          startDate={startDate}
          endDate={endDate}
          eventType={dataPanelCurrentEventType}
          onClose={handleDataPanelClose}
        />
      )}
    </>
  );
}
