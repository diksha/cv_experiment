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
import { Slideover, Dropdown, StickyHeader, DateRangePicker, DateRange, ContentWrapper } from "ui";
import { ReviewTable } from "features/mission-control/review";
import {
  IncidentTypeFilter,
  OrganizationSiteFilter,
  OrganizationSiteFilterOption,
} from "features/mission-control/review";

export const REVIEW_HISTORY_PATH = "/review/history";

export function HistoryPage() {
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [incidentTypeFilter, setIncidentTypeFilter] = useState("all");
  const [organizationSiteFilter, setOrganizationSiteFilter] = useState<OrganizationSiteFilterOption>(
    OrganizationSiteFilter.DefaultValue
  );
  const [externalFeedbackFilter, setExternalFeedbackFilter] = useState("all");
  const [internalFeedbackFilter, setInternalFeedbackFilter] = useState("all");
  const [hasCommentsFilter, setHasCommentsFilter] = useState<any>(null);
  const [dateRangeFilter, setDateRangeFilter] = useState<DateRange>();

  const handleResetFilters = () => {
    setIncidentTypeFilter("all");
    setOrganizationSiteFilter(OrganizationSiteFilter.DefaultValue);
    setExternalFeedbackFilter("all");
    setInternalFeedbackFilter("all");
    setHasCommentsFilter(false);
    setDateRangeFilter(undefined);
    setFiltersOpen(false);
  };

  const filterElements = (
    <>
      <div className="flex flex-wrap flex-col flex-items-center gap-4 md:flex-row">
        <Dropdown
          selectedValue={internalFeedbackFilter}
          options={[
            { value: "all", label: "Reviewer feedback", default: true },
            { value: "valid", label: "Valid" },
            { value: "invalid", label: "Invalid" },
            { value: "unsure", label: "Unsure" },
          ]}
          fetchOnChange={true}
          onChange={(value) => setInternalFeedbackFilter(value)}
          wrapperStyle={styles.filterButton}
          buttonStyle={styles.filterButton}
        />
        <Dropdown
          selectedValue={externalFeedbackFilter}
          options={[
            { value: "all", label: "Customer feedback", default: true },
            { value: "valid", label: "Valid" },
            { value: "invalid", label: "Invalid" },
            { value: "unsure", label: "Unsure" },
          ]}
          fetchOnChange={true}
          onChange={(value) => setExternalFeedbackFilter(value)}
          wrapperStyle={styles.filterButton}
          buttonStyle={styles.filterButton}
        />
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
        <OrganizationSiteFilter
          selectedValues={[organizationSiteFilter]}
          fetchOnChange={true}
          onChange={(value) => setOrganizationSiteFilter(value)}
        />
        <div className="py-2">
          <label className="inline">Comments: </label>
          <input
            type="checkbox"
            checked={hasCommentsFilter}
            onChange={() => setHasCommentsFilter(!hasCommentsFilter)}
          />
        </div>
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
        <title>Review History - Voxel</title>
      </Helmet>
      <TabHeader selectedTab={TabHeader.Tab.History} />
      {responsiveFilters}
      <div className={pageStyles.page}>
        <ReviewTable
          externalFeedbackFilter={externalFeedbackFilter}
          internalFeedbackFilter={internalFeedbackFilter}
          hasCommentsFilter={hasCommentsFilter}
          incidentTypeFilter={incidentTypeFilter}
          organizationSiteFilter={organizationSiteFilter}
          dateRangeFilter={dateRangeFilter}
        />
      </div>
    </>
  );
}
