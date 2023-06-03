import { useLazyQuery } from "@apollo/client";
import { Box, Typography, useMediaQuery, useTheme } from "@mui/material";
import { IncidentTrendBarChart, IncidentType } from "features/dashboard";
import { GET_EXECUTIVE_DASHBOARD_SITE_DATA_PANEL } from "../../queries";
import { DateTime } from "luxon";
import { useEffect, useMemo, useState } from "react";
import { DataPanel, DataPanelSection, DateRange, DateRangePicker, DowntimeBarChartSkeleton, VoxelScoreCard } from "ui";
import { TimeBucketWidth } from "__generated__/globalTypes";
import {
  GetExecutiveDashboardSiteDataPanel,
  GetExecutiveDashboardSiteDataPanelVariables,
} from "__generated__/GetExecutiveDashboardSiteDataPanel";
import { CalendarToday } from "@mui/icons-material";
import { fillIncidentAggregateGroupsMergeIncidents } from "features/dashboard/components/trends/utils";
import { readableDaterange } from "shared/utilities/dateutil";
import { OrgSite, UserSession } from "features/executive-dashboard";
import { filterNullValues } from "shared/utilities/types";
import { VoxelScore } from "shared/types";
import { TrendsByType } from "../trends";
import { MostActiveEmployees } from "../activity";
import { GoToSiteDashboardButton } from "../misc";
import { analytics } from "shared/utilities/analytics";

const COLUMN_FLEX = { display: "flex", flexDirection: "column", gap: 2 };
const ROW_FLEX = { display: "flex", flexDirection: "row", gap: 2 };
interface SiteSummaryDataPanelProps {
  open: boolean;
  startDate: DateTime;
  endDate: DateTime;
  site: OrgSite;
  onClose: () => void;
}
export function SiteSummaryDataPanel({ open, startDate, endDate, site, onClose }: SiteSummaryDataPanelProps) {
  const [dateRangeFilter, setDateRangeFilter] = useState<DateRange>({
    startDate,
    endDate,
  });
  const theme = useTheme();
  const smBreakpoint = useMediaQuery(theme.breakpoints.up("sm"));

  const [getExecutiveDashboardSiteDataPanel, { data, loading }] = useLazyQuery<
    GetExecutiveDashboardSiteDataPanel,
    GetExecutiveDashboardSiteDataPanelVariables
  >(GET_EXECUTIVE_DASHBOARD_SITE_DATA_PANEL, {
    fetchPolicy: "network-only",
    variables: {
      startDate: dateRangeFilter?.startDate?.toISODate(),
      endDate: dateRangeFilter?.endDate?.toISODate(),
      groupBy: TimeBucketWidth.DAY,
      zoneId: site.id,
    },
  });

  useEffect(() => {
    if (open && site) {
      setDateRangeFilter({
        startDate: DateTime.fromISO(startDate.toISODate(), { zone: site.timezone }).startOf("day"),
        endDate: DateTime.fromISO(endDate.toISODate(), { zone: site.timezone }).endOf("day"),
      });
      getExecutiveDashboardSiteDataPanel();
      analytics.trackCustomEvent("viewSitePreview");
    }
  }, [open, site, getExecutiveDashboardSiteDataPanel, startDate, endDate]);

  const handleDateRangeChange = (dateRange: DateRange) => {
    setDateRangeFilter(dateRange);
    analytics.trackCustomEvent("changeDataPanelDateRange");
  };

  const handleClose = () => {
    setDateRangeFilter({
      startDate: DateTime.fromISO(startDate.toISODate(), { zone: site.timezone }).startOf("day"),
      endDate: DateTime.fromISO(endDate.toISODate(), { zone: site.timezone }).endOf("day"),
    });
    onClose();
  };

  const barChart = useMemo(() => {
    if (!dateRangeFilter.startDate || !dateRangeFilter.endDate || !data?.zone) {
      return <></>;
    }
    const incidentTypes = (data?.zone?.incidentTypes as IncidentType[]) || [];

    const trend = fillIncidentAggregateGroupsMergeIncidents(
      [data.zone],
      incidentTypes,
      dateRangeFilter.startDate,
      dateRangeFilter.endDate,
      site.timezone || "",
      false
    )[0];

    const readable = readableDaterange(dateRangeFilter.startDate, dateRangeFilter.endDate, "to") || "";
    const dateTxt = `All Incident Types - ${readable}`;
    return (
      <IncidentTrendBarChart
        data={trend}
        timezone={site.timezone || ""}
        groupBy={TimeBucketWidth.DAY}
        title="Total Incidents"
        secondaryTitle={dateTxt}
        showPercentage
        clickable={false}
      />
    );
  }, [data, dateRangeFilter.startDate, dateRangeFilter.endDate, site]);

  const left = useMemo(() => {
    const sites = data?.zone ? [data.zone] : [];
    const siteScore = data?.zone?.overallScore;
    const siteScoreData = filterNullValues<VoxelScore>(data?.zone?.eventScores);
    const incidentTypes = (data?.zone?.incidentTypes as IncidentType[]) || [];

    return (
      <>
        {siteScore && (
          <DataPanelSection>
            <VoxelScoreCard
              overallScore={siteScore}
              scores={siteScoreData}
              loading={loading}
              title="Voxel Score"
              minWidth={0}
              mode="light"
              boxContainerProps={{}}
              skeletonPadding={0}
            />
          </DataPanelSection>
        )}
        <DataPanelSection>
          <TrendsByType
            sites={sites}
            incidentTypes={incidentTypes}
            startDate={dateRangeFilter.startDate as DateTime}
            endDate={dateRangeFilter.endDate as DateTime}
            loading={loading}
            showChart={false}
            isoDateOnly={false}
            timezone={site.timezone}
            percentageBoxProps={{ paddingLeft: "20px", alignItems: "center" }}
            tableBoxProps={{ sx: { paddingY: 1 } }}
            skeletonPadding={0}
          />
        </DataPanelSection>
      </>
    );
  }, [data, dateRangeFilter.endDate, dateRangeFilter.startDate, loading, site.timezone]);

  const right = useMemo(() => {
    const userSessions = filterNullValues<UserSession>(data?.zone?.sessionCount.users);
    return (
      <>
        <DataPanelSection>
          <MostActiveEmployees
            userSessions={userSessions}
            startDate={dateRangeFilter.startDate as DateTime}
            endDate={dateRangeFilter.endDate as DateTime}
            loading={loading}
            tableBoxProps={{ sx: { paddingY: 1 } }}
          />
        </DataPanelSection>
      </>
    );
  }, [data, dateRangeFilter.endDate, dateRangeFilter.startDate, loading]);

  return (
    <DataPanel open={open} onClose={handleClose}>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }} data-ui-key="datapanel-allsites-site-preview">
        <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap" }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h2">{site.name}</Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: "4px" }}>
            <DateRangePicker
              uiKey="datapanel-date-range-change"
              onChange={handleDateRangeChange}
              values={dateRangeFilter}
              modalStyle="absolute z-30 shadow-lg bg-white md:w-max min-w-screen md:min-w-auto top-16 rounded-2xl right-3"
              alignRight={false}
              alignRightMobile={false}
              icon={<CalendarToday sx={{ height: 16, width: 16 }} />}
              timezone={site.timezone}
              excludedOptions={["Today", "Yesterday", "Custom range"]}
            />
            <GoToSiteDashboardButton site={site} />
          </Box>
        </Box>
        <DataPanelSection>{loading ? <DowntimeBarChartSkeleton /> : barChart}</DataPanelSection>
        {smBreakpoint ? (
          <Box sx={{ ...ROW_FLEX }}>
            <Box sx={{ ...COLUMN_FLEX, flex: "1" }}>{left}</Box>
            <Box sx={{ ...COLUMN_FLEX, flex: "1" }}>{right}</Box>
          </Box>
        ) : (
          <Box sx={{ ...COLUMN_FLEX }}>
            {left}
            {right}
          </Box>
        )}
      </Box>
    </DataPanel>
  );
}
