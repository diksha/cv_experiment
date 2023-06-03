import { DateTime } from "luxon";
import { isEmpty } from "lodash";
import { ReactNode, useMemo } from "react";
import { Helmet } from "react-helmet-async";
import { useQuery } from "@apollo/client";
import { filteredIncidentsLink } from "features/incidents";
import {
  StatisticCard,
  StatisticCardValue,
  SiteScoreCard,
  GET_DASHBOARD_DATA,
  IncidentTrends,
  IncidentsByCameraCard,
  IncidentsByAssigneeCard,
  HighlightedEvents,
  IncidentType,
  Camera,
} from "features/dashboard";
import { useMediaQuery, Box, Typography, useTheme, Paper, Container } from "@mui/material";
import { CurrentZoneActivityFeed } from "features/activity";
import { GetDashboardData, GetDashboardDataVariables } from "__generated__/GetDashboardData";
import { TimeBucketWidth } from "__generated__/globalTypes";
import { useCurrentUser, DOWNTIME_READ } from "features/auth";
import { DashboardProvider } from "../hooks/dashboard";
import { Downtime } from "features/analytics";
import { getNodes } from "graphql/utils";

const COLUMN_FLEX = { display: "flex", flexDirection: "column", gap: 2 };
const ROW_FLEX = { display: "flex", flexDirection: "row", gap: 2 };

export function DashboardPageV3() {
  const theme = useTheme();
  const smBreakpoint = useMediaQuery(theme.breakpoints.up("sm"));
  const mdBreakpoint = useMediaQuery(theme.breakpoints.up("md"));
  const lgBreakpoint = useMediaQuery(theme.breakpoints.up("lg"));
  const { currentUser } = useCurrentUser();
  const timezone = useMemo(() => currentUser?.site?.timezone || DateTime.now().zoneName, [currentUser]);
  const endDate = DateTime.now();
  const endDateString = endDate.toISODate();
  const startDate = endDate.minus({ days: 30 });
  const startDateString = startDate.toISODate();
  const { data, loading } = useQuery<GetDashboardData, GetDashboardDataVariables>(GET_DASHBOARD_DATA, {
    fetchPolicy: "network-only",
    nextFetchPolicy: "cache-and-network",
    variables: {
      startDate: startDateString,
      endDate: endDateString,
      startTimestamp: startDate.startOf("day").toISO(),
      endTimestamp: endDate.endOf("day").toISO(),
      groupBy: TimeBucketWidth.DAY,
    },
  });
  const overallScore = data?.currentUser?.site?.overallScore;
  const eventScores = data?.currentUser?.site?.eventScores;
  const siteScoreEnabled = useMemo(() => {
    return currentUser?.getBooleanPreference("site_score_beta");
  }, [currentUser]);
  const cameras = useMemo(() => {
    return getNodes<Camera>(data?.currentUser?.site?.cameras) || [];
  }, [data]);
  const incidentTypes = useMemo(() => {
    const categories = data?.currentUser?.site?.incidentCategories || [];
    const result: IncidentType[] = [];
    categories.forEach((cat) => {
      if (cat?.incidentTypes) {
        cat.incidentTypes.forEach((t) => {
          if (t) {
            result.push(t);
          }
        });
      }
    });
    return result;
  }, [data]);

  const userStatCards = useMemo(() => {
    return (
      <Box sx={{ ...COLUMN_FLEX }}>
        <Box display="flex" flexDirection={{ xs: "column", md: "row", lg: "column" }} gap={2}>
          <StatisticCard
            title="Assigned to me"
            loading={loading}
            iconVariant="arrowLeft"
            to={filteredIncidentsLink({ ASSIGNMENT: { value: ["ASSIGNED_TO_ME"] } })}
            uiKey="assigned-to-me-card"
          >
            <StatisticCardValue label="Resolved" value={data?.currentUser?.tasksAssignedToStats?.resolvedCount} />
            <StatisticCardValue label="Open" value={data?.currentUser?.tasksAssignedToStats?.openCount} />
          </StatisticCard>
          <StatisticCard
            title="Assigned by me"
            loading={loading}
            iconVariant="arrowRight"
            to={filteredIncidentsLink({ ASSIGNMENT: { value: ["ASSIGNED_BY_ME"] } })}
            uiKey="assigned-by-me-card"
          >
            <StatisticCardValue label="Resolved" value={data?.currentUser?.tasksAssignedByStats?.resolvedCount} />
            <StatisticCardValue label="Open" value={data?.currentUser?.tasksAssignedByStats?.openCount} />
          </StatisticCard>
        </Box>
        <StatisticCard
          title="Bookmarked"
          loading={loading}
          iconVariant="bookmark"
          to={filteredIncidentsLink({ EXTRAS: { value: ["BOOKMARKED"] } })}
          uiKey="bookmarked-card"
        >
          <StatisticCardValue value={data?.currentUser?.stats?.bookmarkTotalCount} />
        </StatisticCard>
      </Box>
    );
  }, [data, loading]);

  const incidentTrends = useMemo(() => {
    const groups = data?.currentUser?.site?.incidentAnalytics?.incidentAggregateGroups || [];
    return (
      <IncidentTrends
        groups={groups}
        incidentTypes={incidentTypes}
        loading={loading}
        startDate={startDate}
        endDate={endDate}
        timezone={timezone}
        cameras={cameras}
      />
    );
  }, [data, loading, startDate, endDate, timezone, cameras, incidentTypes]);

  const incidentStatCards = useMemo(
    () => (
      <Box sx={{ ...COLUMN_FLEX }}>
        <IncidentsByCameraCard
          data={data?.currentUser?.site?.incidentAnalytics?.incidentAggregateGroups}
          incidentTypes={incidentTypes}
          startDate={startDate}
          endDate={endDate}
          timezone={timezone}
        />
        <IncidentsByAssigneeCard
          data={data?.currentUser?.site?.assigneeStats}
          startDate={startDate}
          endDate={endDate}
          timezone={timezone}
        />
      </Box>
    ),
    [data, startDate, endDate, timezone, incidentTypes]
  );

  const downtimeCards = useMemo(() => {
    const productionLines = data?.currentUser?.site?.productionLines || [];
    return (
      <>
        {currentUser?.hasZonePermission(DOWNTIME_READ) && !isEmpty(productionLines) ? (
          <Downtime productionLines={productionLines} timezone={timezone} startDate={startDate} endDate={endDate} />
        ) : null}
      </>
    );
  }, [data, currentUser, timezone, startDate, endDate]);

  let content: ReactNode;

  if (smBreakpoint) {
    let leftColumns: ReactNode;

    const highlightedEvents = <HighlightedEvents viewAllButtonPosition={mdBreakpoint ? "top" : "bottom"} />;

    if (siteScoreEnabled && overallScore && eventScores) {
      leftColumns = (
        <>
          <Box sx={{ ...COLUMN_FLEX }}>
            <SiteScoreCard overallScore={overallScore} eventScores={eventScores} />
            {incidentStatCards}
          </Box>
          <Box sx={{ ...COLUMN_FLEX, flex: "1" }}>
            {highlightedEvents}
            <Box sx={{ ...COLUMN_FLEX }}>
              {downtimeCards}
              {incidentTrends}
              {smBreakpoint && !lgBreakpoint ? userStatCards : null}
            </Box>
          </Box>
        </>
      );
    } else {
      leftColumns = (
        <Box sx={{ ...COLUMN_FLEX, flex: "1" }}>
          {highlightedEvents}
          <Box sx={{ ...ROW_FLEX }}>
            {incidentStatCards}
            <Box sx={{ ...COLUMN_FLEX, flex: "1" }}>
              {downtimeCards}
              {incidentTrends}
              {smBreakpoint && !lgBreakpoint ? userStatCards : null}
            </Box>
          </Box>
        </Box>
      );
    }

    content = (
      <Box sx={{ ...ROW_FLEX }}>
        {leftColumns}
        {lgBreakpoint ? (
          <Box sx={{ ...COLUMN_FLEX, width: "300px" }}>
            {userStatCards}
            <Paper>
              <Box p={2}>
                <Typography variant="h4">Recent Activity</Typography>
                <CurrentZoneActivityFeed />
              </Box>
            </Paper>
          </Box>
        ) : null}
      </Box>
    );
  } else {
    content = (
      <Box sx={{ ...COLUMN_FLEX }}>
        {siteScoreEnabled && overallScore && eventScores ? (
          <SiteScoreCard overallScore={overallScore} eventScores={eventScores} />
        ) : null}
        <HighlightedEvents viewAllButtonPosition="bottom" />
        {downtimeCards}
        {incidentTrends}
        {incidentStatCards}
        {userStatCards}
      </Box>
    );
  }

  return (
    <DashboardProvider>
      <Helmet>
        <title>Dashboard - Voxel</title>
      </Helmet>
      <Container maxWidth="xl" disableGutters sx={{ padding: 2, margin: 0 }}>
        {content}
      </Container>
    </DashboardProvider>
  );
}
