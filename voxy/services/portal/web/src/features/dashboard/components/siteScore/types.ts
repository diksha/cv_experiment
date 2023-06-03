type TierName = "red" | "yellow" | "lightGreen" | "darkGreen";

export type ScoreTier = {
  tier: TierName;
  color: string;
  value: number;
};

export type ScoreTierConfig = ScoreTier[];

export type EventScore = {
  label: string;
  score: number;
};
