import { useRef, useEffect, useMemo, useState } from "react";
import { useQuery } from "@apollo/client";
import classNames from "classnames";
import { useElementSize } from "usehooks-ts";
import { Helmet } from "react-helmet-async";
import { GET_CURRENT_ZONE_INCIDENT_FEED, SkeletonFeedItem, EmptyRangeFeedItem, FeedItemData } from "features/incidents";
import { FilterBar } from "shared/filter/FilterBar";
import { getFilterTypeConfig, useFilters, useDateRangeFilter } from "shared/filter";

import { ContentWrapper, Spinner } from "ui";
import { GetCurrentZoneIncidentFeed } from "__generated__/GetCurrentZoneIncidentFeed";
import { getNodes } from "graphql/utils";
import { DailyIncidentsFeedItem } from "../components";
import { Box, useMediaQuery } from "@mui/material";
import { NAV_HEIGHT_PX } from "ui/layout/constants";

export const INCIDENT_FEED_PATH = "/incidents";
export const DATE_VALUE_FORMAT = "yyyy-MM-dd";

export function IncidentsPage() {
  // TODO PRO-426: remove once backend resolver
  const [timeBucketSizeHours] = useState(24);
  const infiniteScrollRef = useRef(null);
  const { filterBag, serializedFilters } = useFilters();
  const [dateRangeFilter, setDateRangeFilter] = useDateRangeFilter(filterBag);

  const startDate = dateRangeFilter?.startDate ? dateRangeFilter.startDate.toFormat("yyyy-MM-dd") : null;
  const endDate = dateRangeFilter?.endDate ? dateRangeFilter.endDate.toFormat("yyyy-MM-dd") : null;
  const [infiniteScrollLoading, setInfiniteScrollLoading] = useState<boolean>(false);
  const [headerRef, { height: headerHeight }] = useElementSize();
  const mobileBreakpoint = useMediaQuery("(min-width:768px)");

  const { data, loading, fetchMore } = useQuery<GetCurrentZoneIncidentFeed>(GET_CURRENT_ZONE_INCIDENT_FEED, {
    fetchPolicy: "network-only",
    variables: {
      startDate,
      endDate,
      filters: serializedFilters,
      timeBucketSizeHours,
    },
  });

  const feedItems: FeedItemData[] = useMemo(() => {
    return getNodes<FeedItemData>(data?.currentUser?.zone?.incidentFeed);
  }, [data]);

  useEffect(() => {
    const cachedRef = infiniteScrollRef.current;
    if (cachedRef) {
      const observer = new IntersectionObserver(
        ([e]) => {
          if (e.intersectionRatio > 0.0) {
            const pageInfo = data?.currentUser?.zone?.incidentFeed?.pageInfo;
            if (!loading && pageInfo && pageInfo.hasNextPage) {
              setInfiniteScrollLoading(true);
              fetchMore({
                variables: {
                  after: pageInfo.endCursor,
                },
              }).finally(() => {
                setInfiniteScrollLoading(false);
              });
            }
          }
        },
        { threshold: [0.01] }
      );

      observer.observe(cachedRef);

      // unmount
      return function () {
        cachedRef && observer.unobserve(cachedRef!);
      };
    }
  }, [data, fetchMore, loading]);

  const initialLoading = loading && feedItems.length === 0;
  const refetching = loading && !initialLoading && !infiniteScrollLoading;
  const empty = !loading && feedItems.length === 0;

  return (
    <>
      <Helmet>
        <title>Incidents - Voxel</title>
      </Helmet>
      {mobileBreakpoint && <Box paddingTop="16px" />}
      <FilterBar
        ref={headerRef}
        title="Incidents"
        dateRangeFilter={dateRangeFilter}
        setDateRangeFilter={setDateRangeFilter}
        config={getFilterTypeConfig("incidents")}
      />
      <>
        {feedItems.length > 0 && (
          <div className="flex flex-col gap-y-8 md:pb-8">
            {feedItems.map((item) => {
              switch (item.__typename) {
                case "DailyIncidentsFeedItem":
                  return data?.currentUser?.zone?.timezone ? (
                    <DailyIncidentsFeedItem
                      key={item.key}
                      data={item}
                      loading={loading}
                      filters={serializedFilters}
                      stickyTopOffset={headerHeight + NAV_HEIGHT_PX}
                      timezone={data?.currentUser?.zone?.timezone}
                    />
                  ) : null;
                case "EmptyRangeFeedItem":
                  return <EmptyRangeFeedItem key={item.key} data={item} />;
                default:
                  return null;
              }
            })}
            {data?.currentUser?.zone?.incidentFeed?.pageInfo?.hasNextPage ? (
              <ContentWrapper>
                <div ref={infiniteScrollRef}>
                  <SkeletonFeedItem />
                </div>
              </ContentWrapper>
            ) : null}
          </div>
        )}
        {initialLoading && (
          <ContentWrapper>
            <div className="flex flex-col gap-4 md:py-8">
              <SkeletonFeedItem />
              <SkeletonFeedItem />
            </div>
          </ContentWrapper>
        )}
        <div className={classNames("block fixed bottom-16 md:bottom-4 left-0", refetching ? "block" : "hidden")}>
          <div className="w-screen max-w-screen-xl text-center mx-6">
            <div className="inline-block bg-brand-gray-500 text-white px-4 py-3 rounded-lg">
              <div className="flex">
                <Spinner white />
                <div className="pl-2 pt-1 font-bold">Loading...</div>
              </div>
            </div>
          </div>
        </div>
        {empty ? (
          <div className="text-center py-48 md:py-64 text-brand-gray-400 font-bold">
            <div>No incidents.</div>
            <div>Try selecting fewer filters or changing your selected date range.</div>
          </div>
        ) : null}
      </>
    </>
  );
}
