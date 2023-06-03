import { isArray, isEmpty, isString } from "lodash";
import { FilterBag, FilterConfig, FilterValue, SerializedFilter } from ".";
import { DateTime } from "luxon";

const KEYS_EXCLUDED_FROM_FILTER_BAR_COUNT = new Set(["startDate", "endDate"]);

export function getActiveFilterNum(values: FilterBag): number {
  return Object.keys(values || {}).reduce((acc, key) => {
    if (!values[key]?.value) {
      return acc;
    }
    if (KEYS_EXCLUDED_FROM_FILTER_BAR_COUNT.has(key)) {
      return acc;
    }
    switch (typeof values[key]?.value) {
      case "string":
        return (acc = acc + 1);
      default:
        return (acc = acc + (values[key]?.value as string[]).length);
    }
  }, 0);
}

export function isFilterBagEmpty(filterBag: FilterBag): boolean {
  return Object.keys(filterBag).length === 0;
}

export function isFilterValueEmpty(value?: FilterValue): boolean {
  const emptyString = isString(value) && isEmpty(value);
  const emptyArray = isArray(value) && isEmpty(value);
  return !value || emptyString || emptyArray;
}

export function getAWeekAgoStartEndDate(now: DateTime): { startDate: DateTime; endDate: DateTime } {
  return {
    startDate: now.minus({ weeks: 1 }).startOf("week").startOf("day"),
    endDate: now.minus({ weeks: 1 }).endOf("week").endOf("day"),
  };
}

export function serializeFilters(obj: FilterBag): SerializedFilter[] {
  const keys = Object.keys(obj || {});
  return keys.reduce((filters: SerializedFilter[], key) => {
    const value = obj[key]?.value;

    // Exclude filters with "empty" values
    const emptyString = isString(value) && isEmpty(value);
    const emptyArray = isArray(value) && isEmpty(value);
    if (emptyString || emptyArray) {
      return filters;
    }

    // Value looks valid, attempt to serialize it
    // TODO: handle runtime serialization errors, for now let's not fail silently
    return [...filters, { key, valueJson: JSON.stringify(value) }] as SerializedFilter[];
  }, [] as SerializedFilter[]);
}

export function getFilterTypeConfig(type: string): FilterConfig {
  switch (type) {
    case "analytics":
      return {
        allFilters: false,
        incidentTypes: false,
        locations: true,
        assigned: false,
        bookmarked: false,
        highlighted: false,
        clearAll: true,
      };
    case "incidents":
      return {
        allFilters: true,
        incidentTypes: true,
        locations: true,
        assigned: true,
        bookmarked: true,
        highlighted: true,
        clearAll: true,
      };
    default:
      return {
        allFilters: false,
        incidentTypes: false,
        locations: false,
        assigned: false,
        bookmarked: false,
        highlighted: false,
        clearAll: false,
      };
  }
}
