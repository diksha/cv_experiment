import { useMemo } from "react";
import { ProductionLineStatusDataPoint, ChartBrushRange } from "./types";
import { toHumanRead } from "features/analytics/helpers";

interface DowntimeSummaryProps {
  title: string;
  data: ProductionLineStatusDataPoint[];
  brushRange?: ChartBrushRange;
}

function totalDowntimeSeconds(data: ProductionLineStatusDataPoint[], brushRange?: ChartBrushRange): number {
  // NOTE: brush range is inclusive, but .slice() doesn't include the end index, so we add 1 to
  //       ensure the item at the brush end index is included in the sum
  //
  //       slice is tolerant of the end index being out of bounds, so we can safely add 1
  //       regardless of whether the end index is already the max index, slice will just
  //       return up to and including the last item
  const filteredData = brushRange ? data.slice(brushRange.startIndex, brushRange.endIndex + 1) : data;
  const totalSeconds = filteredData.reduce(
    (accumulator, dataPoint) => accumulator + dataPoint.metrics.downtimeDurationSeconds,
    0
  );
  return totalSeconds;
}

export function DowntimeSummary({ title, data, brushRange }: DowntimeSummaryProps) {
  const downtimeDuration = useMemo(() => {
    const totalDurationSeconds = totalDowntimeSeconds(data, brushRange);
    return toHumanRead(totalDurationSeconds).split(",").slice(0, 2).join(" ");
  }, [data, brushRange]);

  if (data.length > 0) {
    return (
      <div className="flex w-full justify-between md:flex-col sm:flex-row">
        <div className="flex-initial">
          <div className="text-base text-brand-blue-900 font-bold">{title}</div>
          <div className="text-xl text-brand-blue-900 font-bold">{downtimeDuration} Downtime</div>
        </div>
      </div>
    );
  } else {
    return (
      <div className="items-center ">
        <div className="py-6 text-base text-brand-blue-900 font-bold">{title}</div>
      </div>
    );
  }
}

export function DowntimeText({ data, brushRange }: DowntimeSummaryProps) {
  const downtimeDuration = useMemo(() => {
    const totalDurationSeconds = totalDowntimeSeconds(data, brushRange);
    return toHumanRead(totalDurationSeconds).split(",").slice(0, 2).join(" ");
  }, [data, brushRange]);

  if (data.length > 0) {
    return <>{downtimeDuration}</>;
  }
  return <>No Data</>;
}
