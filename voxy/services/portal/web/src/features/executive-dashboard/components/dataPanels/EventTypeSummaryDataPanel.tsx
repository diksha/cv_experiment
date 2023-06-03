import { useLazyQuery } from "@apollo/client";
import { Box, Typography, useTheme } from "@mui/material";
import { IncidentTrendBarChart, IncidentTrendSkeleton, IncidentType } from "features/dashboard";
import { GET_EXECUTIVE_DASHBOARD_DATA } from "../../queries";
import { DateTime } from "luxon";
import { useEffect, useMemo, useState } from "react";
import {
  DataPanel,
  DataPanelSection,
  DateRange,
  DateRangePicker,
  DowntimeBarChartSkeleton,
  TrendCard,
  VoxelScoreCard,
} from "ui";
import { TimeBucketWidth } from "__generated__/globalTypes";
import { CalendarToday } from "@mui/icons-material";
import {
  fillIncidentAggregateGroupsMergeIncidents,
  fillIncidentAggregateGroupsMergeSitesIncidents,
} from "features/dashboard/components/trends/utils";
import { GetExecutiveDashboardData, GetExecutiveDashboardDataVariables } from "__generated__/GetExecutiveDashboardData";
import { filterNullValues } from "shared/utilities/types";
import { VoxelScore } from "shared/types";
import { OrgSite } from "features/executive-dashboard";
import { readableDaterange } from "shared/utilities/dateutil";
import { SiteSummaryDataPanel } from "./SiteSummaryDataPanel";
import { FilterBag, serializeFilters } from "shared/filter";
import { analytics } from "shared/utilities/analytics";

interface EventTypeSummaryDataPanelProps {
  open: boolean;
  startDate: DateTime;
  endDate: DateTime;
  eventType: IncidentType;
  onClose: () => void;
}
export function EventTypeSummaryDataPanel({
  open,
  startDate,
  endDate,
  eventType,
  onClose,
}: EventTypeSummaryDataPanelProps) {
  const theme = useTheme();
  const [dateRangeFilter, setDateRangeFilter] = useState<DateRange>({
    startDate,
    endDate,
  });
  const [dataPanelOpen, setDataPanelOpen] = useState(false);
  const [dataPanelCurrentSite, setDataPanelCurrentSite] = useState<OrgSite>();
  const timezone = DateTime.now().zoneName;

  const filterBag: FilterBag = {
    INCIDENT_TYPE: { value: [eventType?.key || ""] },
  };
  const serializedFilters = serializeFilters(filterBag || {});

  const [getExecutiveDashboardData, { data, loading }] = useLazyQuery<
    GetExecutiveDashboardData,
    GetExecutiveDashboardDataVariables
  >(GET_EXECUTIVE_DASHBOARD_DATA, {
    fetchPolicy: "network-only",
    variables: {
      startDate: dateRangeFilter?.startDate?.toISODate(),
      endDate: dateRangeFilter?.endDate?.toISODate(),
      groupBy: TimeBucketWidth.DAY,
      filters: serializedFilters,
    },
  });

  useEffect(() => {
    if (open && eventType?.key) {
      setDateRangeFilter({
        startDate,
        endDate,
      });
      getExecutiveDashboardData();
      analytics.trackCustomEvent("dataPanelIncidentPreview");
    }
  }, [open, eventType?.key, getExecutiveDashboardData, startDate, endDate]);

  const sites = useMemo(() => {
    return filterNullValues<OrgSite>(data?.currentUser?.organization?.sites).filter((site) => site.isActive);
  }, [data]);

  const siteEventScoreData = useMemo(() => {
    return filterNullValues<VoxelScore>(
      sites.map((site) => {
        const scoreValue = site?.eventScores?.find((score) => score?.label === eventType?.name)?.value;
        if (typeof scoreValue === "number" && site) {
          return {
            label: site.name,
            value: scoreValue,
          };
        }
        return null;
      })
    );
  }, [sites, eventType?.name]);

  const eventScore = useMemo(() => {
    return data?.currentUser?.organization?.eventScores?.find((score) => score && score.label === eventType?.name);
  }, [data, eventType?.name]);

  const barChart = useMemo(() => {
    if (!dateRangeFilter.startDate || !dateRangeFilter.endDate || !data || !eventType) {
      return <></>;
    }
    const mergedMap = fillIncidentAggregateGroupsMergeSitesIncidents(
      sites,
      [eventType],
      dateRangeFilter.startDate,
      dateRangeFilter.endDate,
      timezone,
      true
    );
    const trend = mergedMap[eventType.key];
    const readable = readableDaterange(dateRangeFilter.startDate, dateRangeFilter.endDate, "to") || "";
    const dateTxt = `Total Incidents - All Sites - ${readable}`;
    return (
      <IncidentTrendBarChart
        data={trend}
        timezone={timezone || ""}
        groupBy={TimeBucketWidth.DAY}
        title={trend.countTotal.toString()}
        secondaryTitle={dateTxt}
        showPercentage
        clickable={false}
      />
    );
  }, [data, dateRangeFilter.startDate, dateRangeFilter.endDate, eventType, sites, timezone]);

  const trendCards = useMemo(() => {
    if (!dateRangeFilter.startDate || !dateRangeFilter.endDate || !data || !eventType) {
      return <></>;
    }
    const trends = fillIncidentAggregateGroupsMergeIncidents(
      sites,
      [eventType],
      dateRangeFilter.startDate,
      dateRangeFilter.endDate,
      timezone,
      true
    );
    trends.sort((a, b) => b.countTotal - a.countTotal);
    const readable = readableDaterange(startDate, endDate, "to") || "";
    const dateText = `Total Incidents - ${readable}`;
    return trends.map((trend) => {
      const site = sites.find((s) => s.name === trend.name) as OrgSite;
      return (
        <TrendCard
          key={trend.name}
          trend={trend}
          title={trend.name}
          secondaryTitle={dateText}
          paperProps={{ sx: { border: `1px solid ${theme.palette.grey[300]}`, cursor: "pointer" } }}
          onClick={() => handleClick(site)}
        />
      );
    });
  }, [
    sites,
    startDate,
    endDate,
    timezone,
    dateRangeFilter.startDate,
    dateRangeFilter.endDate,
    eventType,
    theme.palette.grey,
    data,
  ]);

  const handleDateRangeChange = (dateRange: DateRange) => {
    analytics.trackCustomEvent("changeDataPanelDateRange");
    setDateRangeFilter(dateRange);
  };

  const handleClose = () => {
    setDateRangeFilter({
      startDate,
      endDate,
    });
    onClose();
  };

  const handleClick = (site: OrgSite) => {
    setDataPanelOpen(true);
    setDataPanelCurrentSite(site);
  };

  const handleDataPanelClose = () => {
    setDataPanelOpen(false);
    setDataPanelCurrentSite(undefined);
  };

  return (
    <DataPanel open={open} onClose={handleClose}>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }} data-ui-key="datapanel-allsites-incident-preview">
        <Box sx={{ display: "flex", alignItems: "center" }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h2">{eventType?.name}</Typography>
          </Box>
          <Box>
            <DateRangePicker
              uiKey="datapanel-date-range-change"
              onChange={handleDateRangeChange}
              values={dateRangeFilter}
              modalStyle="absolute z-30 shadow-lg bg-white md:w-max min-w-screen md:min-w-auto top-16 rounded-2xl right-3"
              alignRight={false}
              alignRightMobile={false}
              icon={<CalendarToday sx={{ height: 16, width: 16 }} />}
              timezone={timezone}
              excludedOptions={["Today", "Yesterday", "Custom range"]}
            />
          </Box>
        </Box>
        <VoxelScoreCard overallScore={eventScore} scores={siteEventScoreData} loading={loading} title="Voxel Score" />
        <DataPanelSection>{loading ? <DowntimeBarChartSkeleton /> : barChart}</DataPanelSection>
        {loading ? (
          <>
            <IncidentTrendSkeleton />
            <IncidentTrendSkeleton />
            <IncidentTrendSkeleton />
          </>
        ) : (
          trendCards
        )}
        {!!dataPanelCurrentSite && (
          <SiteSummaryDataPanel
            open={dataPanelOpen}
            startDate={dateRangeFilter.startDate as DateTime}
            endDate={dateRangeFilter.endDate as DateTime}
            site={dataPanelCurrentSite}
            onClose={handleDataPanelClose}
          />
        )}
      </Box>
    </DataPanel>
  );
}
