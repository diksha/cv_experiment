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
import React, { useState, useEffect, createRef } from "react";
import { LineChart as RechartsLineChart, Line, Tooltip, CartesianGrid, XAxis, YAxis } from "recharts";
import { Series, ObjectMapping } from "shared/types";
import { IncidentType } from "features/analytics";

export type PriorityType = "lowPriorityCount" | "mediumPriorityCount" | "highPriorityCount";

export interface PriorityOption {
  label: string;
  backgroundColor: string;
  value: PriorityType;
}

interface ChartProps {
  height: number;
  width: number;
  data: Series[];
  chartType: string;
  incidentTypes?: IncidentType[];
  priorities?: PriorityOption[];
}

/**
 * Wrapper around Rechart LineChart with dynamic width.
 * TODO(troycarlson): Add support for resizing.
 */
export const LineChart = ({ height, width, data, chartType, incidentTypes, priorities }: ChartProps) => {
  const wrapperRef = createRef<HTMLDivElement>();
  const [incidentTypeMap, setIncidentTypeMap] = useState<ObjectMapping>({});
  const [priorityMap, setPriorityMap] = useState<ObjectMapping>({});

  useEffect(() => {
    let _incidentTypeMap: ObjectMapping = {};
    if (incidentTypes) {
      for (const item of incidentTypes) {
        _incidentTypeMap[item.key] = item;
      }
    }
    setIncidentTypeMap(_incidentTypeMap);
  }, [incidentTypes]);

  useEffect(() => {
    let _priorityMap: ObjectMapping = {};
    if (priorities) {
      for (const item of priorities) {
        _priorityMap[item.value] = item;
      }
    }
    setPriorityMap(_priorityMap);
  }, [priorities]);

  const renderIncidentTypeLines = () => {
    const lines = incidentTypes
      ?.map((item) => item.key)
      .map((key) => (
        <Line
          key={key}
          type="monotone"
          name={incidentTypeMap[key]?.name}
          dataKey={`incidentTypeCounts.${key}`}
          stroke={incidentTypeMap[key]?.backgroundColor}
          strokeWidth={3}
          dot={false}
        />
      ));
    return lines;
  };

  const renderPriorityLines = () => {
    const lines = priorities
      ?.map((item) => item.value)
      .map((key) => (
        <Line
          key={key}
          type="monotone"
          name={`${priorityMap[key]?.label}`}
          dataKey={`priorityCounts.${priorityMap[key]?.value}`}
          stroke={`${priorityMap[key]?.backgroundColor}`}
          strokeWidth={3}
          dot={false}
        />
      ));
    return lines;
  };

  return (
    <div ref={wrapperRef}>
      <RechartsLineChart width={width} height={height} data={data} margin={{ top: 5, right: 30, bottom: 5, left: 5 }}>
        <XAxis dataKey="key" />
        <YAxis allowDecimals={false} />
        <CartesianGrid strokeDasharray="3 3" />
        <Tooltip />
        {chartType === "incidentType" && renderIncidentTypeLines()}
        {chartType === "incidentPriority" && renderPriorityLines()}
      </RechartsLineChart>
    </div>
  );
};
