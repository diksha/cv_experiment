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
import { DateTime, DateTimeFormatOptions } from "luxon";

export const dateFormatOpts: DateTimeFormatOptions = {
  month: "short",
  day: "numeric",
};

type ShortcutDateRange = {
  startDate: DateTime;
  endDate: DateTime;
};

export type Shortcut = {
  label: (timezone: string) => string;
  range?: (timezone: string) => ShortcutDateRange;
};

function getLocalizedNow(timezone: string): DateTime {
  return DateTime.now().setZone(timezone, { keepLocalTime: true });
}

function getRangeForLastNDays(days: number, timezone: string): ShortcutDateRange {
  const localizedNow = getLocalizedNow(timezone);
  return {
    startDate: localizedNow.minus({ days }).startOf("day"),
    endDate: localizedNow.endOf("day"),
  };
}

function getRangeForLastNMonths(months: number, timezone: string): ShortcutDateRange {
  const localizedNow = getLocalizedNow(timezone);
  return {
    startDate: localizedNow.minus({ months }).startOf("day"),
    endDate: localizedNow.endOf("day"),
  };
}

export const shortcuts: Shortcut[] = [
  {
    label: () => "Today",
    range: (timezone) => ({
      startDate: getLocalizedNow(timezone).startOf("day"),
      endDate: getLocalizedNow(timezone).endOf("day"),
    }),
  },
  {
    label: () => "Yesterday",
    range: (timezone) => ({
      startDate: getLocalizedNow(timezone).minus({ days: 1 }).startOf("day"),
      endDate: getLocalizedNow(timezone).minus({ days: 1 }).endOf("day"),
    }),
  },
  {
    label: () => "Last 7 Days",
    range: (timezone) => getRangeForLastNDays(7, timezone),
  },
  {
    label: () => "Last 14 Days",
    range: (timezone) => getRangeForLastNDays(14, timezone),
  },
  {
    label: () => "Last 30 Days",
    range: (timezone) => getRangeForLastNDays(30, timezone),
  },
  {
    // Last month
    label: (timezone) => {
      const monthAgo = getLocalizedNow(timezone).minus({ months: 1 });
      return `${monthAgo.monthLong} ${monthAgo.year}`;
    },
    range: (timezone) => ({
      startDate: getLocalizedNow(timezone).minus({ months: 1 }).startOf("month"),
      endDate: getLocalizedNow(timezone).minus({ months: 1 }).endOf("month"),
    }),
  },
  {
    label: () => "Last 60 Days",
    range: (timezone) => getRangeForLastNDays(60, timezone),
  },
  {
    label: () => "Last 90 Days",
    range: (timezone) => getRangeForLastNDays(90, timezone),
  },
  {
    label: () => "Last 6 Months",
    range: (timezone) => getRangeForLastNMonths(6, timezone),
  },
  {
    label: () => "Last 12 Months",
    range: (timezone) => getRangeForLastNMonths(12, timezone),
  },
];

export function date2string(date?: DateTime, format?: DateTimeFormatOptions): string | null {
  return date ? date.toLocaleString(format) : null;
}
