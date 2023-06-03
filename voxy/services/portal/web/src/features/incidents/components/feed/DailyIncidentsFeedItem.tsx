/*
 * Copyright 2020-2021 Voxel Labs, Inc.
 * All rights reserved.
 *
 * This document may not be reproduced, republished, distributed, transmitted,
 * displayed, broadcast or otherwise exploited in any manner without the express
 * prior written permission of Voxel Labs, Inc. The receipt or possession of this
 * document does not convey any rights to reproduce, disclose, or distribute its
 * contents, or to manufacture, use, or sell anything that it may describe, in
 * whole or in part.
 */
import { relativeDateString } from "shared/utilities/dateutil";
import { DateTime } from "luxon";
import { getNodes } from "graphql/utils";
import { ContentWrapper, StickyHeader, Spinner, useLocalizedDateTime } from "ui";
import React, { useEffect, useMemo, useState, useRef } from "react";
import { useLazyQuery } from "@apollo/client";
import {
  IncidentRow,
  GET_INCIDENT_FEED,
  DailyIncidentsFeedItemData,
  DailyIncidentsFeedItemTimeBucketData,
} from "features/incidents";
import classNames from "classnames";
import { GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets_incidentCounts } from "__generated__/GetCurrentZoneIncidentFeed";
import {
  GetIncidentFeed,
  GetIncidentFeedVariables,
  GetIncidentFeed_incidentFeed_edges_node,
} from "__generated__/GetIncidentFeed";
import { SerializedFilter } from "shared/filter";

interface IncidentTypeCountData
  extends GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets_incidentCounts {}

function timeBucketTitle(date: DateTime, title: string): string {
  return relativeDateString(date, "MMM d");
}

export function DailyIncidentsFeedItem(props: {
  data: DailyIncidentsFeedItemData;
  filters?: SerializedFilter[];
  loading: boolean;
  stickyTopOffset: number;
  timezone: string;
}) {
  const { data } = props;
  const date = useLocalizedDateTime(data.date, true);

  const timeBuckets = useMemo(() => {
    // For now, don't render buckets without data
    const bucketsWithData = data.timeBuckets.filter((timeBucket) => timeBucket && timeBucket?.incidentCount > 0);
    // Reverse the list to produce reverse chronological ordering
    return [...bucketsWithData].reverse();
  }, [data]);

  return (
    <>
      {timeBuckets.map((timeBucket) =>
        timeBucket ? (
          <TimeBucket
            key={timeBucket.key}
            data={timeBucket}
            date={date}
            filters={props.filters}
            loading={props.loading}
            stickyTopOffset={props.stickyTopOffset}
          />
        ) : null
      )}
    </>
  );
}

function TimeBucket(props: {
  data: DailyIncidentsFeedItemTimeBucketData;
  date: DateTime;
  filters?: SerializedFilter[];
  loading: boolean;
  stickyTopOffset: number;
}) {
  const [activeRow, setActiveRow] = useState<number | string | null>(null);
  const [headerStuck, setHeaderStuck] = useState(false);
  const [, { loading, data, fetchMore }] = useLazyQuery<GetIncidentFeed, GetIncidentFeedVariables>(GET_INCIDENT_FEED, {
    fetchPolicy: "network-only",
    nextFetchPolicy: "cache-first",
    notifyOnNetworkStatusChange: true,
    variables: {
      first: (props.data.latestIncidents?.length || 0) + 10,
      // TODO: move start/end to standard filters
      fromUtc: props.data.startTimestamp,
      toUtc: props.data.endTimestamp,
      filters: props.filters,
    },
  });

  const ref = useRef(null);
  const [refRegistered, setRefRegistered] = useState(false);

  const incidents: GetIncidentFeed_incidentFeed_edges_node[] = useMemo(() => {
    if (!data) {
      return props.data.latestIncidents || [];
    }
    // TODO: replace `any` with proper type
    return getNodes<any>(data.incidentFeed);
  }, [props.data.latestIncidents, data]);

  const remainingCount = props.data.incidentCount - incidents.length;
  const allowShowMore = remainingCount > 0;

  const handleShowMore = () => {
    fetchMore({
      variables: {
        after: data?.incidentFeed?.pageInfo.endCursor,
      },
    });
  };

  const handleRowClick = (id: string | number | null) => {
    if (id === activeRow || !id) {
      // Collapse row
      setActiveRow(null);
    } else {
      // Expand row
      setActiveRow(id);
    }
  };

  useEffect(() => {
    if (ref?.current && !refRegistered) {
      setRefRegistered(true);
    }
  }, [ref, refRegistered, props]);

  const title = useMemo(() => {
    return timeBucketTitle(props.date, props.data.title);
  }, [props.date, props.data]);

  const handleStuck = () => {
    setHeaderStuck(true);
  };
  const handleUnstuck = () => {
    setHeaderStuck(false);
  };

  const stickyHeaderClasses = classNames("transition-colors duration-75", {
    "bg-white border-b border-brand-gray-050": headerStuck,
  });

  const headerClasses = classNames("overflow-hidden", {
    "border-b border-brand-gray-050": !headerStuck,
  });

  return (
    <div ref={ref} data-time-bucket-key={props.data.key}>
      <StickyHeader
        className={stickyHeaderClasses}
        top={props.stickyTopOffset}
        zIndex={30}
        onStuck={handleStuck}
        onUnStuck={handleUnstuck}
      >
        <ContentWrapper>
          <div className={headerClasses}>
            <TimeBucketHeader data={props.data.incidentCounts} title={title} totalCount={props.data.incidentCount} />
          </div>
        </ContentWrapper>
      </StickyHeader>

      <ContentWrapper>
        <div
          className={classNames("bg-white md:rounded-br-lg md:rounded-bl-lg overflow-hidden", {
            "opacity-75": props.loading,
          })}
        >
          {incidents.map((incident: GetIncidentFeed_incidentFeed_edges_node) => (
            <IncidentRow
              key={incident.id}
              incident={incident}
              active={activeRow === incident.id}
              onClick={() => handleRowClick(incident.id || null)}
            />
          ))}
          {allowShowMore && !props.loading ? (
            <button
              className="grid justify-center w-full bg-white text-sm border-t py-4 px-4 hover:bg-gray-50"
              onClick={handleShowMore}
            >
              {loading ? (
                <div className="grid opacity-50">
                  <Spinner />
                </div>
              ) : (
                <span>+ {remainingCount} more</span>
              )}
            </button>
          ) : null}
        </div>
      </ContentWrapper>
    </div>
  );
}

export function TimeBucketHeader(props: {
  title: string;
  data: (IncidentTypeCountData | null)[] | null;
  totalCount: number;
}) {
  return (
    <div className={classNames("md:flex justify-between items-center bg-white z-30 gap-2")}>
      <div
        className={classNames(
          "p-4 flex md:grid justify-between content-center align-self-start",
          "font-bold text-medium text-brand-gray-500"
        )}
      >
        <div className="whitespace-nowrap">{props.title}</div>
        <div className="md:hidden font-bold text-brand-gray-500">{props.totalCount}</div>
      </div>
      <div
        className={classNames(
          "flex bg-gray-200 p-2 md:rounded-lg overflow-x-scroll scrollbar-hidden whitespace-nowrap"
        )}
      >
        {props.data?.map((incidentCount) => {
          return (
            <div className={classNames("mx-4 flex items-center")} key={incidentCount?.incidentType?.key}>
              <div className={classNames("bg-gray-400 text-xs text-white rounded-full h-5 w-5 leading-5 text-center")}>
                {incidentCount?.count}
              </div>
              <p className={classNames("mx-2 text-xs")}>{incidentCount?.incidentType?.name}</p>
            </div>
          );
        })}
      </div>
      <div className="hidden md:block font-bold text-brand-gray-500 px-4">{props.totalCount}</div>
    </div>
  );
}
