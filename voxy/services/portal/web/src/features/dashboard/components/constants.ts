import { ColorConfig } from "shared/types";

export const YELLOW_SCORE_THRESHOLD = 40;
export const GREEN_SCORE_THRESHOLD = 80;
export const MAX_HR_DAYS = 4;
export const MIN_TREND_INCIDENT_COUNT = 50;
export const MIN_TREND_DAYS = 28;
export const PERIOD_WEEK = 7;
export const PERIOD_MONTH = 28;
export const HIDDEN_TREND_INCIDENT_TYPES: Record<string, string> = {
  PRODUCTION_LINE_DOWN: "PRODUCTION_LINE_DOWN",
};
export const CAMERA_COLORS: ColorConfig[] = [
  { fill: "#810F7C", text: "#ffffff" },
  { fill: "#225EA8", text: "#ffffff" },
  { fill: "#DF65B0", text: "#ffffff" },
  { fill: "#455A64", text: "#ffffff" },
  { fill: "#4D004B", text: "#ffffff" },
  { fill: "#C51B7D", text: "#ffffff" },
  { fill: "#5CC7FA", text: "#ffffff" },
  { fill: "#263238", text: "#ffffff" },
  { fill: "#8C6BB1", text: "#ffffff" },
  { fill: "#D9F0A3", text: "#000000" },
  { fill: "#1D91C0", text: "#ffffff" },
  { fill: "#DFC27D", text: "#000000" },
  { fill: "#F2C966", text: "#000000" },
  { fill: "#37474F", text: "#ffffff" },
  { fill: "#B27832", text: "#000000" },
  { fill: "#546E7A", text: "#ffffff" },
  { fill: "#FEE08B", text: "#000000" },
  { fill: "#607D8B", text: "#ffffff" },
  { fill: "#8C510A", text: "#000000" },
  { fill: "#78909C", text: "#ffffff" },
  { fill: "#00000A", text: "#ffffff" },
  { fill: "#010114", text: "#ffffff" },
  { fill: "#01011F", text: "#ffffff" },
  { fill: "#020229", text: "#ffffff" },
  { fill: "#020233", text: "#ffffff" },
  { fill: "#35355C", text: "#ffffff" },
  { fill: "#676785", text: "#ffffff" },
  { fill: "#9A9AAD", text: "#ffffff" },
  { fill: "#CCCCD6", text: "#ffffff" },
];
export const INCIDENT_TYPE_TEXT_COLORS: Record<string, string> = {
  DOOR_VIOLATION: "#ffffff",
  OPEN_DOOR_DURATION: "#ffffff",
  SPILL: "#ffffff",
  NO_STOP_AT_INTERSECTION: "#ffffff",
  PARKING_DURATION: "#ffffff",
  PIGGYBACK: "#ffffff",
  NO_STOP_AT_END_OF_AISLE: "#ffffff",
  NO_STOP_AT_DOOR_INTERSECTION: "#ffffff",
  HARD_HAT: "#000000",
  NO_PED_ZONE: "#000000",
  OVERREACHING: "#000000",
  BAD_POSTURE: "#000000",
  Safety_Harness: "#000000",
  SAFETY_VEST: "#000000",
};
