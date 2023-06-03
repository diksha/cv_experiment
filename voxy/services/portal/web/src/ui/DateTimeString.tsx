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
import { useCurrentUser } from "features/auth";
import { localizedDateTime } from "shared/utilities/dateutil";

export function useLocalizedDateTime(dateTime: string | DateTime, keepLocalTime: boolean = false) {
  const { currentUser } = useCurrentUser();
  return localizedDateTime(dateTime, keepLocalTime, currentUser);
}

export function DateTimeString(props: { dateTime: string | DateTime; format?: string; includeTimezone?: boolean }) {
  const formatUsed = props.format ? props.format : "LLL d y @ h:mm";
  const dt = useLocalizedDateTime(props.dateTime);

  return (
    <span>
      {dt.toFormat(formatUsed)}
      <span>{dt.toFormat("a").toLowerCase()}</span>
      {props.includeTimezone ? <span>{dt.toFormat(" ZZZZ")}</span> : null}
    </span>
  );
}

export function TimeString(props: { dateTime: string | DateTime }) {
  const dt = typeof props.dateTime === "string" ? DateTime.fromISO(props.dateTime) : props.dateTime;

  return (
    <span>
      {dt.toFormat("h:mm")}
      <span>{dt.toFormat("a").toLowerCase()}</span>
      <span>{dt.toFormat(" ZZZZ")}</span>
    </span>
  );
}

export function formatDateTimeString(dateTime: string | DateTime, tz?: string) {
  const dt = typeof dateTime === "string" ? DateTime.fromISO(dateTime).setZone(tz) : dateTime.setZone(tz);

  const dateAndTime = dt.toFormat("LLL d y @ h:mm");
  const ampm = dt.toFormat("a").toLowerCase();
  const timezone = dt.zoneName;
  return `${dateAndTime}${ampm} ${timezone}`;
}

export function getDateTimeRangeString(startDateTime: string | DateTime, endDateTime: DateTime, tz?: string): string {
  const startDT =
    typeof startDateTime === "string" ? DateTime.fromISO(startDateTime).setZone(tz) : startDateTime.setZone(tz);

  return `${startDT.toFormat("EEE, MMM d")} from ${getTimeRangeString(startDateTime, endDateTime, tz)}`;
}

export function getTimeRangeString(startDateTime: string | DateTime, endDateTime: DateTime, tz?: string): string {
  const startDT =
    typeof startDateTime === "string" ? DateTime.fromISO(startDateTime).setZone(tz) : startDateTime.setZone(tz);
  const endDT = typeof endDateTime === "string" ? DateTime.fromISO(endDateTime).setZone(tz) : endDateTime.setZone(tz);

  const startString = startDT.toLocaleString(DateTime.TIME_SIMPLE);
  const endString = endDT.toLocaleString(DateTime.TIME_SIMPLE);
  const timeZone = startDT.toFormat("ZZZZ");

  return `${startString} ${timeZone} to ${endString} ${timeZone}`;
}
