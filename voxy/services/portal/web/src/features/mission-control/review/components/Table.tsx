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
import { useEffect, useState, useRef } from "react";
import classNames from "classnames";
import { DateTime } from "luxon";
import { useLazyQuery } from "@apollo/client";
import { GET_INCIDENT_FEEDBACK_FEED } from "features/incidents";
import { CaretUp, CaretDown } from "phosphor-react";
import { Transition } from "@headlessui/react";
import { IncidentVideo, OrganizationSiteFilterOption } from "features/mission-control/review";
import { Spinner, DateRange } from "ui";
import {
  GetIncidentFeedbackSummary,
  GetIncidentFeedbackSummaryVariables,
  GetIncidentFeedbackSummary_incidentFeedbackSummary_edges,
  GetIncidentFeedbackSummary_incidentFeedbackSummary_edges_node,
} from "__generated__/GetIncidentFeedbackSummary";
const defaultEmptyMessage = "No incidents match your filters";

interface ReviewTableProps {
  externalFeedbackFilter: string;
  internalFeedbackFilter: string;
  hasCommentsFilter: boolean | null;
  incidentTypeFilter: string;
  organizationSiteFilter: OrganizationSiteFilterOption;
  dateRangeFilter?: DateRange;
}

export function ReviewTable({
  externalFeedbackFilter,
  internalFeedbackFilter,
  hasCommentsFilter,
  incidentTypeFilter,
  organizationSiteFilter,
  dateRangeFilter,
}: ReviewTableProps) {
  const ref = useRef(null);
  const [filtersReady, setFiltersReady] = useState(false);
  const [activeRow, setActiveRow] = useState<string | null>(null);
  const [getFeedbackFeed, { loading, data, fetchMore, refetch }] = useLazyQuery<
    GetIncidentFeedbackSummary,
    GetIncidentFeedbackSummaryVariables
  >(GET_INCIDENT_FEEDBACK_FEED, {
    fetchPolicy: "network-only",
    nextFetchPolicy: "cache-first",
    variables: {
      first: 20,
      externalFeedback: externalFeedbackFilter,
      internalFeedback: internalFeedbackFilter,
      hasComments: hasCommentsFilter,
      incidentType: incidentTypeFilter,
      organizationId: organizationSiteFilter.organizationId,
      siteId: organizationSiteFilter.siteId,
      fromUtc: dateRangeFilter?.startDate,
      toUtc: dateRangeFilter?.endDate,
    },
  });
  const showSpinner = loading || (data && data?.incidentFeedbackSummary?.pageInfo.hasNextPage);
  const edgesIsEmpty = data?.incidentFeedbackSummary?.edges.length === 0;
  const showEmpty = !loading && edgesIsEmpty;
  const showData = !loading && !edgesIsEmpty;

  useEffect(() => {
    refetch && refetch();
    setFiltersReady(true);
  }, [
    externalFeedbackFilter,
    internalFeedbackFilter,
    hasCommentsFilter,
    incidentTypeFilter,
    organizationSiteFilter,
    dateRangeFilter,
    refetch,
  ]);

  useEffect(() => {
    // Initial fetch, filters must be flagged as "ready"
    filtersReady && getFeedbackFeed();
  }, [filtersReady, getFeedbackFeed]);

  useEffect(() => {
    const cachedRef = ref.current;
    if (cachedRef) {
      const observer = new IntersectionObserver(
        ([e]) => {
          if (e.intersectionRatio > 0.5) {
            if (showData) {
              fetchMore &&
                fetchMore({
                  variables: {
                    after: data?.incidentFeedbackSummary?.pageInfo.endCursor,
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

  const spinnerClasses = classNames("grid justify-center opacity-40", showData ? "p-8" : "p-36");

  return (
    <>
      {loading && (
        <div className={spinnerClasses}>
          <Spinner />
        </div>
      )}
      {!loading && (
        <div className="flex flex-col">
          <div className="-my-2 overflow-x-auto sm:-mx-6 lg:-mx-8">
            <div className="py-2 align-middle w-full inline-block min-w-full sm:px-6 lg:px-8">
              <div className="overflow-hidden border border-gray-200 sm:rounded-lg">
                <table className="w-full min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th
                        scope="col"
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                      >
                        Last activity
                      </th>
                      <th
                        scope="col"
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider hidden lg:table-cell"
                      >
                        Valid
                      </th>
                      <th
                        scope="col"
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider hidden lg:table-cell"
                      >
                        Invalid
                      </th>
                      <th
                        scope="col"
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider hidden lg:table-cell"
                      >
                        Unsure
                      </th>
                      <th
                        scope="col"
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider lg:hidden"
                      >
                        Feedback
                      </th>
                      <th scope="col" className="relative px-6 py-3">
                        <span className="sr-only">View</span>
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {showData && (
                      <Rows
                        edges={data?.incidentFeedbackSummary?.edges || []}
                        activeRow={activeRow}
                        setActiveRow={setActiveRow}
                      />
                    )}
                  </tbody>
                </table>
                {showEmpty ? (
                  <div className="grid justify-center p-36 bg-white">
                    <div className="opacity-40">{defaultEmptyMessage}</div>
                  </div>
                ) : null}
                {showSpinner ? (
                  <div className={spinnerClasses} ref={ref}>
                    <div>
                      <Spinner />
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function Rows(props: {
  edges: (GetIncidentFeedbackSummary_incidentFeedbackSummary_edges | null)[];
  activeRow: string | null;
  setActiveRow: (id: string | null) => void;
}): JSX.Element {
  const { edges, activeRow, setActiveRow } = props;
  const handleRowClick = (id: string | null) => {
    if (id === activeRow || !id) {
      // Collapse row
      setActiveRow(null);
    } else {
      // Expand row
      setActiveRow(id);
    }
  };
  return (
    <>
      {edges.length > 0
        ? edges.map((edge, idx): JSX.Element => {
            if (edge?.node) {
              return (
                <Row
                  key={idx}
                  incidentId={edge.node.id}
                  incidentUuid={edge.node.uuid}
                  row={edge.node}
                  expanded={activeRow === edge.node.id}
                  onClick={() => handleRowClick(edge.node?.id || null)}
                />
              );
            }
            return <></>;
          })
        : null}
    </>
  );
}

function Row(props: {
  incidentId: string;
  incidentUuid: string;
  row: GetIncidentFeedbackSummary_incidentFeedbackSummary_edges_node;
  expanded: boolean;
  onClick: (incidentId: string) => void;
}) {
  const { row, incidentId, incidentUuid, expanded, onClick } = props;
  const bgClass =
    row.unsure.length > 0 || (row.valid.length > 0 && row.invalid.length > 0) ? "bg-yellow-100" : "bg-white";

  return (
    <>
      <tr
        key={incidentId}
        // Highlight incidents marked as unsure or with conflicting valid/invalid feedback
        className={classNames("cursor-pointer", bgClass)}
        style={{ borderBottom: expanded ? "hidden" : "" }}
        onClick={() => onClick(incidentId)}
      >
        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
          {DateTime.fromISO(row.lastFeedbackSubmissionTimestamp).toRelative()}
        </td>
        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 hidden lg:table-cell">
          {row.valid.map((x: string | null) => (
            <div key={x}>{x}</div>
          ))}
        </td>
        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 hidden lg:table-cell">
          {row.invalid.map((x: string | null) => (
            <div key={x}>{x}</div>
          ))}
        </td>
        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 hidden lg:table-cell">
          {row.unsure.map((x: string | null) => (
            <div key={x}>{x}</div>
          ))}
        </td>
        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 lg:hidden">
          {row.valid.length > 0 ? (
            <div className="pb-4">
              <div className="font-bold">Valid</div>
              {row.valid.map((x: string | null) => (
                <div key={x}>{x}</div>
              ))}
            </div>
          ) : null}
          {row.invalid.length > 0 ? (
            <div className="pb-4">
              <div className="font-bold">Invalid</div>
              {row.invalid.map((x: string | null) => (
                <div key={x}>{x}</div>
              ))}
            </div>
          ) : null}
          {row.unsure.length > 0 ? (
            <div className="pb-4">
              <div className="font-bold">Unsure</div>
              {row.unsure.map((x: string | null) => (
                <div key={x}>{x}</div>
              ))}
            </div>
          ) : null}
        </td>
        <td className="px-6 py-4 whitespace-nowrap">{expanded ? <CaretUp size={18} /> : <CaretDown size={18} />}</td>
      </tr>
      <tr className={classNames("!border-0", bgClass)}>
        <Transition
          show={expanded}
          as="td"
          enter="transition-all ease duration-100"
          enterFrom="transform opacity-0 scale-y-0"
          enterTo="transform opacity-100 scale-y-100"
          className={classNames("!border-0", bgClass)}
          colSpan={5}
        >
          <div className="p-4">
            <IncidentVideo incidentId={incidentId} incidentUuid={incidentUuid} />
          </div>
        </Transition>
      </tr>
    </>
  );
}
