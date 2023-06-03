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
import React, { useState } from "react";
import { Helmet } from "react-helmet-async";
import pageStyles from "shared/styles/pages.module.css";
import styles from "./ReviewPage.module.css";
import { Faders } from "phosphor-react";
import { TabHeader } from "../components";
import { ContentWrapper, Slideover, StickyHeader, DateRangePicker, DateRange } from "ui";
import { Table } from "features/incidents";
import { IncidentTypeFilter } from "features/mission-control/review";

export const REVIEW_EXPERIMENTS_PATH = "/review/experiments";

export function ExperimentsPage() {
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [incidentTypeFilter, setIncidentTypeFilter] = useState("all");
  const [dateRangeFilter, setDateRangeFilter] = useState<DateRange>();

  const handleResetFilters = () => {
    setIncidentTypeFilter("all");
    setDateRangeFilter(undefined);
    setFiltersOpen(false);
  };

  const filterElements = (
    <>
      <div className="flex flex-col flex-items-center gap-4 md:flex-row">
        <DateRangePicker
          placeholder="Date range"
          onChange={(dateRange) => setDateRangeFilter(dateRange)}
          values={dateRangeFilter}
          wrapperStyle={styles.filterButton}
          buttonStyle={styles.filterButton}
          buttonActiveStyle=""
          buttonSelectedStyle=""
          buttonItemsStyle=""
          modalStyle=""
          overrideBaseStyle={false}
          alignRight={false}
          alignRightMobile={false}
          forceMobileLayout={false}
          closeOnSelection={false}
        />
        <IncidentTypeFilter
          selectedValues={[incidentTypeFilter]}
          fetchOnChange={true}
          onChange={(value) => setIncidentTypeFilter(value)}
        />
        <button type="button" className="text-gray-400 inline-block text-left" onClick={handleResetFilters}>
          Reset
        </button>
      </div>
    </>
  );

  const responsiveFilters = (
    <>
      <div className="inline md:hidden">
        <button className="fixed bottom-16 right-2 z-30">
          <div
            className={classNames(
              "flex gap-2 py-2 px-4 items-center shadow-lg",
              "rounded-full bg-brand-blue-900 text-white"
            )}
            onClick={() => setFiltersOpen(true)}
          >
            <Faders size={16} />
            <p>Filters</p>
          </div>
        </button>
        <Slideover title="Filters" open={filtersOpen} onClose={() => setFiltersOpen(false)}>
          {filterElements}
        </Slideover>
      </div>
      <StickyHeader className="bg-white border-b border-brand-gray-050 py-4 hidden md:block" zIndex={30}>
        <ContentWrapper>{filterElements}</ContentWrapper>
      </StickyHeader>
    </>
  );

  return (
    <>
      <Helmet>
        <title>Review Experiments - Voxel</title>
      </Helmet>
      <TabHeader selectedTab={TabHeader.Tab.Experiments} />
      {responsiveFilters}
      <div className={pageStyles.page}>
        <Table
          dateRangeFilter={dateRangeFilter}
          incidentTypeFilter={[incidentTypeFilter]}
          experimentalFilter={true}
          filtersReady={true}
        />
      </div>
    </>
  );
}
