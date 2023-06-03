import { useSearchParams } from "react-router-dom";
import { useCallback, useEffect } from "react";
import { useQuery } from "@apollo/client";
import { getFilterTypeConfig, useFilters, useDateRangeFilter, FilterBar } from "shared/filter";
import { DateTime } from "luxon";
import { filterNullValues } from "shared/utilities/types";
import { Helmet } from "react-helmet-async";
import {
  GetAnalyticsPageData,
  GetAnalyticsPageData_analytics_series,
  GetAnalyticsPageDataVariables,
  GetAnalyticsPageData_currentUser_site_incidentTypes,
} from "__generated__/GetAnalyticsPageData";
import { useCurrentUser } from "features/auth";
import { GET_ANALYTICS_PAGE_DATA, IncidentChart } from "features/analytics";
import { TimeUnit, TimeUnitEnum } from "../types/time"; // Direct import from a file instead of the index to avoid compiling error
import { Card, DateRange, PageTitle, useLocalizedDateTime } from "ui";
import { IncidentStatistics } from "../components";
import { Container, useMediaQuery } from "@mui/material";

interface SeriesDataPoint extends GetAnalyticsPageData_analytics_series {}

export const ANALYTICS_PATH = "/analytics";

const TimeUnitMap: Record<string, TimeUnit> = {
  hour: {
    singular: TimeUnitEnum.Hour,
    plural: "hours",
    groupBy: TimeUnitEnum.Hour,
    chartKeyFormat: "ha",
    tickFormat: "LLL d, yyyy ZZZZ",
  },
  day: {
    singular: TimeUnitEnum.Day,
    plural: "days",
    groupBy: TimeUnitEnum.Hour,
    chartKeyFormat: "ha ZZZZ",
    tickFormat: "ccc h:mma",
  },
  week: {
    singular: TimeUnitEnum.Week,
    plural: "weeks",
    groupBy: TimeUnitEnum.Day,
    chartKeyFormat: "ccc, LLL d",
    tickFormat: "LLL d, yyyy h:mma",
  },
  month: {
    singular: TimeUnitEnum.Month,
    plural: "months",
    groupBy: TimeUnitEnum.Day,
    chartKeyFormat: "LL/d",
    tickFormat: "LLL d, yyyy",
  },
  year: {
    singular: TimeUnitEnum.Year,
    plural: "years",
    groupBy: TimeUnitEnum.Month,
    chartKeyFormat: "LLL",
    tickFormat: "LLLL d",
  },
};

function getXAxisUnit(startDate: DateTime, endDate: DateTime): TimeUnit {
  const timeDeltaDays = Math.round(endDate.diff(startDate, "days").days);
  if (timeDeltaDays < 2) {
    return TimeUnitMap[TimeUnitEnum.Day];
  } else if (timeDeltaDays <= 7) {
    return TimeUnitMap[TimeUnitEnum.Week];
  } else if (timeDeltaDays <= 31) {
    return TimeUnitMap[TimeUnitEnum.Month];
  } else {
    return TimeUnitMap[TimeUnitEnum.Year];
  }
}

export function AnalyticsPage() {
  const { currentUser } = useCurrentUser();
  const timezone = currentUser?.site?.timezone || DateTime.now().zoneName;
  const [searchParams, setSearchParams] = useSearchParams();
  const { filterBag, serializedFilters } = useFilters();
  // Default date range is last 30 days
  const defaultStartDate = DateTime.now().setZone(timezone).minus({ days: 30 }).startOf("day");
  const defaultEndDate = DateTime.now().setZone(timezone).endOf("day");
  const defaultDateRangeFilter = { startDate: defaultStartDate, endDate: defaultEndDate };
  const [dateRangeFilter, setDateRangeFilter] = useDateRangeFilter(filterBag);
  const startDate = dateRangeFilter?.startDate || defaultStartDate;
  const endDate = dateRangeFilter?.endDate || defaultEndDate;
  const xAxisUnit = getXAxisUnit(startDate, endDate);
  const mobileBreakpoint = useMediaQuery("(min-width:768px)"); // TODO(hq): some places use tailwind breakpoints (md: 768), others use mui breakpoints (sm 600). should just use one.

  const { data, loading } = useQuery<GetAnalyticsPageData, GetAnalyticsPageDataVariables>(GET_ANALYTICS_PAGE_DATA, {
    fetchPolicy: "network-only",
    // Used for subsequent executions
    nextFetchPolicy: "cache-first",
    variables: {
      startTimestamp: useLocalizedDateTime(startDate, true).toISO(),
      endTimestamp: useLocalizedDateTime(endDate, true).toISO(),
      groupBy: xAxisUnit.groupBy,
      filters: serializedFilters,
    },
  });

  const incidentTypes = filterNullValues<GetAnalyticsPageData_currentUser_site_incidentTypes>(
    data?.currentUser?.site?.incidentTypes
  );

  const series =
    data?.analytics?.series
      ?.slice()
      .filter((point): point is SeriesDataPoint => point !== null)
      .sort((a: any, b: any) => new Date(a.key).getTime() - new Date(b.key).getTime())
      .map((point) => {
        return {
          ...point,
          // Redefine key with formatted value based on selected time unit
          key: DateTime.fromISO(point.key, { zone: timezone }).toFormat(xAxisUnit.chartKeyFormat),
        };
      }) || [];

  const updateQueryParam = useCallback(
    (name: string, value: string) => {
      const params = Object.fromEntries(searchParams);
      params[name] = value;
      setSearchParams(params, { replace: true });
    },
    [searchParams, setSearchParams]
  );

  const handleDateRangeChange = useCallback(
    (dateRange: DateRange) => {
      const { startDate, endDate } = dateRange;
      if (startDate && endDate) {
        updateQueryParam("fromUtc", startDate.toISO());
        updateQueryParam("toUtc", endDate.toISO());
        setDateRangeFilter({
          startDate: startDate,
          endDate: endDate,
        });
      }
    },
    [setDateRangeFilter, updateQueryParam]
  );

  useEffect(() => {
    const startDate = searchParams.get("fromUtc");
    const endDate = searchParams.get("toUtc");
    let dateRange: DateRange | undefined;
    if (startDate && endDate) {
      dateRange = { startDate: DateTime.fromISO(startDate), endDate: DateTime.fromISO(endDate) };
      handleDateRangeChange(dateRange);
    }
  }, [handleDateRangeChange, searchParams]);

  return (
    <>
      <Helmet>
        <title>Analytics - Voxel</title>
      </Helmet>
      <Container maxWidth="xl" disableGutters sx={{ padding: 2, margin: 0 }}>
        {mobileBreakpoint && (
          <PageTitle
            title="Analytics"
            secondaryTitle="Explore incident trends across your site"
            boxPadding="0 0 30px"
          />
        )}
        <Card noPadding>
          <IncidentChart loading={loading} series={series} incidentTypes={incidentTypes}>
            <FilterBar
              dateRangeFilter={dateRangeFilter || defaultDateRangeFilter}
              defaultDateRangeFilter={defaultDateRangeFilter}
              setDateRangeFilter={setDateRangeFilter}
              config={getFilterTypeConfig("analytics")}
              // Hide reset option until we have a more robust date picker solution
              showDatePickerResetButton={false}
              sticky={false}
            />
          </IncidentChart>
          <IncidentStatistics
            startDate={startDate}
            endDate={endDate}
            incidentTypes={incidentTypes}
            loading={loading}
            series={series}
          />
        </Card>
      </Container>
    </>
  );
}
