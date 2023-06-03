/*
 * Copyright 2022 Voxel Labs, Inc.
 * All rights reserved.
 *
 * This document may not be reproduced, republished, distributed, transmitted,
 * displayed, broadcast or otherwise exploited in any manner without the express
 * prior written permission of Voxel Labs, Inc. The receipt or possession of this
 * document does not convey any rights to reproduce, disclose, or distribute its
 * contents, or to manufacture, use, or sell anything that it may describe, in
 * whole or in part.
 */

import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import {
  SerializedFilter,
  FilterBag,
  ParseFiltersResponse,
  SerializeFiltersResponse,
  FilterHookResponse,
} from "./types";
import { getAWeekAgoStartEndDate, isFilterValueEmpty } from "./helpers";
import { DateRange } from "ui";
import { isEmpty, isString } from "lodash";
import { useCurrentUser } from "features/auth";
import { localizedDateTime } from "shared/utilities/dateutil";
import { DateTime } from "luxon";

function serializeFilters(filterBag: FilterBag): SerializeFiltersResponse {
  const searchParams = new URLSearchParams();
  const serializedFilters: SerializedFilter[] = [];
  for (const [key, value] of Object.entries(filterBag)) {
    // Exclude filters with "empty" values
    if (!value || isFilterValueEmpty(value)) {
      continue;
    }

    // Value looks valid, attempt to serialize it
    // TODO: handle runtime serialization errors, for now let's not fail silently
    const valueJson = JSON.stringify(value.value);
    searchParams.append(key, valueJson);
    serializedFilters.push({ key, valueJson });
  }

  return { searchParams, serializedFilters };
}

function parseFilters(searchParams: URLSearchParams): ParseFiltersResponse {
  const filterBag: FilterBag = {};
  const serializedFilters: SerializedFilter[] = [];
  const searchParamsObject = Object.fromEntries(searchParams.entries());

  for (const [key, valueJson] of Object.entries(searchParamsObject)) {
    filterBag[key] = { value: JSON.parse(valueJson) };
    serializedFilters.push({ key, valueJson });
  }

  return { filterBag, serializedFilters };
}

/**
 * React hook which manages filter state and syncs filter state with URL search params.
 *
 * @returns {FilterHookResponse} Hook response object
 */
export function useFilters(): FilterHookResponse {
  const [searchParams, setSearchParams] = useSearchParams();
  const [initialFilters] = useState<ParseFiltersResponse>(() => parseFilters(searchParams));
  const [desiredFilterBag, setDesiredFilterBag] = useState<FilterBag>(() => initialFilters.filterBag);
  const [currentFilterBag, setCurrentFilterBag] = useState<FilterBag>(desiredFilterBag);
  const [serializedFilters, setSerializedFilters] = useState<SerializedFilter[]>(
    () => initialFilters.serializedFilters
  );

  useEffect(() => {
    if (JSON.stringify(desiredFilterBag) !== JSON.stringify(currentFilterBag)) {
      const { serializedFilters: newSerializedFilters, searchParams: newSearchParams } =
        serializeFilters(desiredFilterBag);
      setSerializedFilters(newSerializedFilters);
      setSearchParams(newSearchParams, { replace: true });
      setCurrentFilterBag(desiredFilterBag);
    }
  }, [desiredFilterBag, currentFilterBag, searchParams, setSearchParams]);

  useEffect(() => {
    const { filterBag: newFilterBag, serializedFilters: newSerializedFilters } = parseFilters(searchParams);
    if (JSON.stringify(newFilterBag) !== JSON.stringify(currentFilterBag)) {
      setSerializedFilters(newSerializedFilters);
      setDesiredFilterBag(newFilterBag);
      setCurrentFilterBag(newFilterBag);
    }
  }, [currentFilterBag, searchParams, setSearchParams]);

  return { filterBag: currentFilterBag, serializedFilters, setFilters: setDesiredFilterBag };
}

export function useDateRangeFilter(filterBag: FilterBag, useDefault: boolean = false) {
  const { currentUser } = useCurrentUser();

  // When the start date and end date are not provided,
  // and the does not want to use the default date time range, return undefined instead.
  // The incident page uses this case.
  let result: DateRange | undefined = undefined;
  const { startDate, endDate } = filterBag;

  // When the start date and end date are not provided, and would like to use the default date time range.
  if (useDefault && isEmpty(startDate) && isEmpty(endDate)) {
    const { startDate, endDate } = getAWeekAgoStartEndDate(DateTime.now());
    result = {
      startDate: localizedDateTime(startDate, true, currentUser),
      endDate: localizedDateTime(endDate, true, currentUser),
    };
  }

  // When the start date and end date is provided from the filter bag
  // convert the date string format "2022-01-01",
  // to date time string format "2023-01-01T13:00:00.000-05:00.
  if (!isEmpty(startDate) && !isEmpty(endDate) && isString(startDate.value) && isString(endDate.value)) {
    result = {
      startDate: localizedDateTime(startDate.value, true, currentUser).startOf("day"),
      endDate: localizedDateTime(endDate.value, true, currentUser).endOf("day"),
    };
  }

  return useState<DateRange | undefined>(() => result);
}
