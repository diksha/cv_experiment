import { Duration } from "luxon";

export function toHumanRead(seconds: number) {
  const duration = Duration.fromObject({ seconds: seconds });
  return duration.rescale().toHuman({ unitDisplay: "short" });
}
