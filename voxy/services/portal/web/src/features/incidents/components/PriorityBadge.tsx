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
import React from "react";
import classNames from "classnames";

interface StringMap {
  [key: string]: string;
}

const badgeStyles: StringMap = {
  high: "text-brand-red-500",
  medium: "text-brand-orange-500",
  low: "text-brand-yellow-500",
  resolved: "text-brand-green-600",
  unknown: "text-gray-600",
};

const sizeStyles: StringMap = {
  small: "text-xs",
  normal: "text-sm",
};

const badgeLabels: StringMap = {
  high: "HIGH",
  medium: "MEDIUM",
  low: "LOW",
  resolved: "RESOLVED",
  unknown: "UNKNOWN",
};

export function PriorityBadge(props: { priority: string; status: string; small?: boolean }) {
  const badgeSize = props.small ? "small" : "normal";
  let badgeType = "unknown";
  const priority = props.priority.toLowerCase();
  const status = props.status.toLowerCase();
  if (status === "resolved") {
    badgeType = "resolved";
  } else if (badgeStyles[priority]) {
    badgeType = priority;
  }

  const badgeClasses = classNames(
    "p-2 inline-block text-center font-bold",
    sizeStyles[badgeSize],
    badgeStyles[badgeType]
  );

  return (
    <div className="inline-block">
      <div className={badgeClasses}>{badgeLabels[badgeType]}</div>
    </div>
  );
}
