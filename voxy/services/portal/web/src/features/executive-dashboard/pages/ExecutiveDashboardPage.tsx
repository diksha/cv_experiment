import { Helmet } from "react-helmet-async";
import { useQuery } from "@apollo/client";
import { Box, Container, Typography, useMediaQuery, useTheme } from "@mui/material";
import { DateRange, DateRangePicker, VoxelScoreCard } from "ui";
import { TimeBucketWidth } from "__generated__/globalTypes";
import { GetExecutiveDashboardData, GetExecutiveDashboardDataVariables } from "__generated__/GetExecutiveDashboardData";
import { GET_EXECUTIVE_DASHBOARD_DATA } from "../queries";
import { ScoresByType, SiteSummaryDataPanel, TrendsBySite, TrendsByType } from "../components";
import { DateTime } from "luxon";
import { OrgSite, SiteSession, UserSession } from "../types";
import { filterNullValues } from "shared/utilities/types";
import { useCallback, useMemo, useState } from "react";
import { CalendarToday } from "@mui/icons-material";
import { VoxelScore } from "shared/types";
import { MostActiveEmployees, MostActiveSites } from "../components/activity";
import { DownloadReportButton } from "../components/pdf";
import { analytics } from "shared/utilities/analytics";

const COLUMN_FLEX = { display: "flex", flexDirection: "column", gap: 2 };
const ROW_FLEX = { display: "flex", flexDirection: "row", gap: 2 };

export function ExecutiveDashboardPage() {
  const timezone = DateTime.now().zoneName;
  const theme = useTheme();
  const mdBreakpoint = useMediaQuery(theme.breakpoints.up("md"));
  const [dateRangeFilter, setDateRangeFilter] = useState<DateRange>({
    startDate: DateTime.now().startOf("day").minus({ days: 30 }),
    endDate: DateTime.now().endOf("day"),
  });
  const [sitedataPanelOpen, setSiteDataPanelOpen] = useState(false);
  const [siteDataPanelCurrentSite, setSiteDataPanelCurrentSite] = useState<OrgSite>();

  const { data, loading } = useQuery<GetExecutiveDashboardData, GetExecutiveDashboardDataVariables>(
    GET_EXECUTIVE_DASHBOARD_DATA,
    {
      fetchPolicy: "network-only",
      nextFetchPolicy: "cache-and-network",
      variables: {
        startDate: dateRangeFilter?.startDate?.toISODate(),
        endDate: dateRangeFilter?.endDate?.toISODate(),
        groupBy: TimeBucketWidth.DAY,
      },
    }
  );

  const sites = useMemo(() => {
    return filterNullValues<OrgSite>(data?.currentUser?.organization?.sites).filter((site) => site.isActive);
  }, [data]);
  const incidentTypes = useMemo(() => {
    return data?.currentUser?.organization?.incidentTypes || [];
  }, [data]);

  const handleDateRangeChange = (dateRange: DateRange) => {
    analytics.trackCustomEvent("changeDashboardDateRange");
    setDateRangeFilter(dateRange);
  };

  const handleSiteClick = useCallback(
    (score: VoxelScore) => {
      const site = sites.find((s) => s.name === score.label) as OrgSite;
      setSiteDataPanelOpen(true);
      setSiteDataPanelCurrentSite(site);
    },
    [sites]
  );

  const handleSiteDataPanelClose = () => {
    setSiteDataPanelOpen(false);
    setSiteDataPanelCurrentSite(undefined);
  };

  const left = useMemo(() => {
    const siteScore = data?.currentUser?.organization?.overallScore;
    const siteScoreData = filterNullValues<VoxelScore>(
      sites.map((site) => {
        return site?.overallScore;
      })
    );
    const eventScoreData = filterNullValues<VoxelScore>(data?.currentUser?.organization?.eventScores);
    return (
      <>
        <VoxelScoreCard
          overallScore={siteScore}
          scores={siteScoreData}
          loading={loading}
          title="Voxel Score"
          onClick={handleSiteClick}
        />
        <ScoresByType
          scores={eventScoreData}
          incidentTypes={incidentTypes}
          startDate={dateRangeFilter.startDate as DateTime}
          endDate={dateRangeFilter.endDate as DateTime}
          loading={loading}
        />
      </>
    );
  }, [data, sites, dateRangeFilter.endDate, dateRangeFilter.startDate, incidentTypes, loading, handleSiteClick]);
  const middle = useMemo(() => {
    return (
      <>
        <TrendsBySite
          sites={sites}
          incidentTypes={incidentTypes}
          startDate={dateRangeFilter.startDate as DateTime}
          endDate={dateRangeFilter.endDate as DateTime}
          loading={loading}
        />
        <TrendsByType
          sites={sites}
          incidentTypes={incidentTypes}
          startDate={dateRangeFilter.startDate as DateTime}
          endDate={dateRangeFilter.endDate as DateTime}
          loading={loading}
          clickable
        />
      </>
    );
  }, [dateRangeFilter.endDate, dateRangeFilter.startDate, incidentTypes, loading, sites]);
  const right = useMemo(() => {
    const siteSessions = filterNullValues<SiteSession>(data?.currentUser?.organization?.sessionCount.sites).filter(
      (session) => session.site?.isActive
    );
    const userSessions = filterNullValues<UserSession>(data?.currentUser?.organization?.sessionCount.users);
    return (
      <>
        <MostActiveSites
          siteSessions={siteSessions}
          sites={sites}
          startDate={dateRangeFilter.startDate as DateTime}
          endDate={dateRangeFilter.endDate as DateTime}
          loading={loading}
        />
        <MostActiveEmployees
          userSessions={userSessions}
          startDate={dateRangeFilter.startDate as DateTime}
          endDate={dateRangeFilter.endDate as DateTime}
          loading={loading}
          clickable
          showHeader
          showIcon
          showPagination
        />
      </>
    );
  }, [data, dateRangeFilter.endDate, dateRangeFilter.startDate, loading, sites]);

  return (
    <>
      <Helmet>
        <title>Executive Dashboard - Voxel</title>
      </Helmet>
      <Container maxWidth="xl" disableGutters sx={{ padding: "1rem", margin: 0 }}>
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexGrow: "1",
            height: "58px",
            backgroundColor: theme.palette.common.white,
            padding: "0 32px",
            marginBottom: "16px",
            borderRadius: "8px",
          }}
        >
          {mdBreakpoint && <Typography variant="h4">All Sites Dashboard</Typography>}
          <Box sx={{ display: "flex", alignItems: "center" }}>
            <DateRangePicker
              uiKey="dashboard-allsites-change-date-range"
              onChange={handleDateRangeChange}
              values={dateRangeFilter}
              alignRight={false}
              alignRightMobile={false}
              icon={<CalendarToday sx={{ height: 16, width: 16 }} />}
              timezone={timezone}
              excludedOptions={["Today", "Yesterday", "Custom range"]}
            />
            <Box marginLeft={1}>
              <DownloadReportButton
                data={data}
                startDate={dateRangeFilter.startDate as DateTime}
                endDate={dateRangeFilter.endDate as DateTime}
              />
            </Box>
          </Box>
        </Box>
        {mdBreakpoint ? (
          <Box sx={{ ...ROW_FLEX }}>
            <Box sx={{ ...COLUMN_FLEX }}>{left}</Box>
            <Box sx={{ ...COLUMN_FLEX, flex: "1" }}>{middle}</Box>
            <Box sx={{ ...COLUMN_FLEX, width: "360px" }}>{right}</Box>
          </Box>
        ) : (
          <Box sx={{ ...COLUMN_FLEX }}>
            {left}
            {middle}
            {right}
          </Box>
        )}
      </Container>
      {!!siteDataPanelCurrentSite && (
        <SiteSummaryDataPanel
          open={sitedataPanelOpen}
          startDate={dateRangeFilter.startDate as DateTime}
          endDate={dateRangeFilter.endDate as DateTime}
          site={siteDataPanelCurrentSite}
          onClose={handleSiteDataPanelClose}
        />
      )}
    </>
  );
}
