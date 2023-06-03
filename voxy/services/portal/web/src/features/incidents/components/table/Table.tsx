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
import classNames from "classnames";
import { useEffect, useRef, useState } from "react";
import { useLazyQuery } from "@apollo/client";
import { TableRow, GET_INCIDENT_FEED } from "features/incidents";
import { SerializedFilter } from "shared/filter";
import { Card, Spinner, DateRange } from "ui";
import {
  GetIncidentFeed,
  GetIncidentFeedVariables,
  GetIncidentFeed_incidentFeed_edges,
} from "__generated__/GetIncidentFeed";

const defaultEmptyMessage = "No incidents match your filters";

interface Props {
  emptyMessage?: string;
  filters?: SerializedFilter[];
  // Old filters
  dateRangeFilter?: DateRange;
  priorityFilter?: string[];
  incidentTypeFilter?: string[];
  statusFilter?: string[];
  cameraFilter?: string[];
  listFilter?: string[];
  assigneeFilter?: string[];
  experimentalFilter?: boolean;
  filtersReady: boolean;
}

export function Table({
  emptyMessage,
  filters,
  // Old filters
  dateRangeFilter,
  priorityFilter,
  incidentTypeFilter,
  statusFilter,
  cameraFilter,
  listFilter,
  assigneeFilter,
  experimentalFilter,
  filtersReady,
}: Props) {
  const [activeRow, setActiveRow] = useState<number | string | null>(null);
  const ref = useRef(null);
  const { startDate, endDate } = dateRangeFilter || {};
  const [getIncidentFeed, { loading, data, fetchMore, refetch }] = useLazyQuery<
    GetIncidentFeed,
    GetIncidentFeedVariables
  >(GET_INCIDENT_FEED, {
    // Used for first execution
    fetchPolicy: "network-only",
    // Used for subsequent executions
    nextFetchPolicy: "cache-first",
    variables: {
      first: 20,
      fromUtc: startDate,
      toUtc: endDate,
      filters,
      // Old filters
      priorityFilter,
      incidentTypeFilter,
      statusFilter,
      cameraFilter,
      listFilter,
      assigneeFilter,
      experimentalFilter,
    },
  });

  const edges = data?.incidentFeed?.edges || [];
  const isEmpty = edges.length === 0;
  const showSpinner = loading || (data && data?.incidentFeed?.pageInfo.hasNextPage);
  const showEmpty = !loading && isEmpty;
  const showData = !loading && !isEmpty;

  useEffect(() => {
    refetch && refetch();
  }, [
    dateRangeFilter,
    priorityFilter,
    incidentTypeFilter,
    statusFilter,
    cameraFilter,
    listFilter,
    assigneeFilter,
    refetch,
  ]);

  useEffect(() => {
    // Initial fetch, filters must be flagged as "ready"
    filtersReady && getIncidentFeed();
  }, [filtersReady, getIncidentFeed]);

  useEffect(() => {
    const cachedRef = ref.current;
    if (cachedRef) {
      const observer = new IntersectionObserver(
        ([e]) => {
          if (e.intersectionRatio > 0.5) {
            // If incidents are already present, execute fetchMore()
            if (showData) {
              fetchMore &&
                fetchMore({
                  variables: {
                    after: data?.incidentFeed?.pageInfo.endCursor,
                  },
                });
            }
          }
        },
        { threshold: [1] }
      );

      observer.observe(cachedRef);

      // unmount
      return function () {
        cachedRef && observer.unobserve(cachedRef!);
      };
    }
  }, [data, fetchMore, loading, showData]);

  const handleRowClick = (id: string | number | null) => {
    if (id === activeRow || !id) {
      // Collapse row
      setActiveRow(null);
    } else {
      // Expand row
      setActiveRow(id);
    }
  };

  const spinnerClasses = classNames("grid justify-center opacity-40", showData ? "p-8" : "p-36");

  return (
    <Card noPadding>
      {showData && (
        <div>
          {edges.map((edge: GetIncidentFeed_incidentFeed_edges | null) => {
            if (edge?.node) {
              return (
                <TableRow
                  key={`incident-row-${edge.node.id}`}
                  incident={edge.node}
                  active={activeRow === edge.node.id}
                  onClick={() => handleRowClick(edge.node?.id!)}
                />
              );
            }
            return null;
          })}
        </div>
      )}
      {showEmpty ? (
        <div className="grid justify-center opacity-40 p-36">{emptyMessage ? emptyMessage : defaultEmptyMessage}</div>
      ) : null}
      {showSpinner ? (
        <div className={spinnerClasses} ref={ref}>
          <div>
            <Spinner />
          </div>
        </div>
      ) : null}
    </Card>
  );
}
