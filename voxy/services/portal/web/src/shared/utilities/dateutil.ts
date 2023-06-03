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
import { DateTime } from "luxon";
import { CurrentUser } from "features/auth";
import { date2string, dateFormatOpts } from "ui/datepicker/utils";

export function isToday(value: DateTime): boolean {
  return value.hasSame(DateTime.local(), "day");
}

export function isYesterday(value: DateTime): boolean {
  return value.hasSame(DateTime.local().minus({ days: 1 }), "day");
}

export function relativeDateString(date: DateTime, format: string = "MMM d"): string {
  return isToday(date) ? "Today" : isYesterday(date) ? "Yesterday" : date.toFormat(format);
}

export function localizedDateTime(
  dateTime: string | DateTime,
  keepLocalTime: boolean = false,
  currentUser: CurrentUser | undefined
): DateTime {
  return typeof dateTime === "string"
    ? DateTime.fromISO(dateTime).setZone(currentUser?.site?.timezone, { keepLocalTime })
    : dateTime.setZone(currentUser?.site?.timezone, { keepLocalTime });
}

export function readableDaterange(startDate: DateTime, endDate: DateTime, separator: string = "-") {
  if (startDate.hasSame(endDate, "day")) {
    return date2string(startDate, dateFormatOpts);
  } else if (startDate.hasSame(endDate, "month")) {
    return `${date2string(startDate, dateFormatOpts)} ${separator} ${date2string(endDate, { day: "numeric" })}`;
  }
  return `${date2string(startDate, dateFormatOpts)} ${separator} ${date2string(endDate, dateFormatOpts)}`;
}
