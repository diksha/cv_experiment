import { ScoreTier, ScoreTierConfig } from "./types";

export function getScoreTier(score: number, config: ScoreTierConfig): ScoreTier {
  for (let i = config.length - 1; i >= 0; i--) {
    if (score >= config[i].value) {
      return config[i];
    }
  }
  return config[0];
}
