import { ScoreTierConfig } from "./types";

export const BAR_BACKGROUND_COLOR = "#2a2f49";
export const BAR_BACKGROUND_COLOR_GREY = "#e9e9ec";

export const defaultScoreTierConfig: ScoreTierConfig = [
  {
    tier: "red",
    color: "#d84315",
    value: 0,
  },
  {
    tier: "yellow",
    color: "#ffc107",
    value: 25,
  },
  {
    tier: "lightGreen",
    color: "#86BA27",
    value: 50,
  },
  {
    tier: "darkGreen",
    color: "#017f28",
    value: 75,
  },
];
