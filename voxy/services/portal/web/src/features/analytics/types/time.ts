import { DateTime } from "luxon";

export type DateRangeFilter = {
  startDate: DateTime;
  endDate: DateTime;
};

// Direct import from a file instead of the index to avoid compiling error
export enum TimeUnitEnum {
  Hour = "hour",
  Day = "day",
  Week = "week",
  Month = "month",
  Year = "year",
}

export type TimeUnit = {
  singular: TimeUnitEnum;
  plural: string;
  groupBy: TimeUnitEnum;
  chartKeyFormat: string;
  tickFormat: string;
};
