import { startCase, camelCase } from "lodash";

import { DateTime } from "luxon";

export function toTitleCase(str: string): string {
  return startCase(camelCase(str));
}

export function truncateString(str: string, num: number): string {
  if (str.length <= num) {
    return str;
  }
  return str.slice(0, num) + "...";
}

export function toDatetimeFormat(label: string | number, format: string = "LLL d, yyyy h:mma"): string {
  if (typeof label === "string") {
    return DateTime.fromISO(label, { setZone: true }).toFormat(format);
  }
  return label.toString();
}
