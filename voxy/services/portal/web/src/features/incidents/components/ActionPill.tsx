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
import { CircleWavyCheck, Eye, Fire, ArrowBendDownRight, Warning, CheckCircle } from "phosphor-react";
import { Tooltip } from "@mui/material";

export enum PillActions {
  RESOLVED,
  HIGH_PRIORITY,
  MEDIUM_PRIORITY,
  LOW_PRIORITY,
  ALERTED,
  ASSIGNED,
  SEEN,
  HIGHLIGHTED,
}

type StringMap = {
  [key in PillActions]: string;
};

type IconMap = {
  [key in PillActions]: () => React.ReactNode;
};

const pillStyles: StringMap = {
  [PillActions.ASSIGNED]: "bg-white border border-brand-gray-100 text-brand-gray-500",
  [PillActions.RESOLVED]: "bg-white border border-brand-green-100 text-brand-green-700",
  [PillActions.ALERTED]: "bg-white border border-brand-purple-100 text-brand-purple-500",
  [PillActions.HIGHLIGHTED]: "bg-white border border-brand-purple-100 text-brand-purple-500",
  [PillActions.HIGH_PRIORITY]: "bg-white border border-brand-red-100 text-brand-red-600",
  [PillActions.MEDIUM_PRIORITY]: "bg-white border border-brand-orange-100 text-brand-orange-500",
  [PillActions.LOW_PRIORITY]: "bg-white border border-brand-yellow-100 text-brand-yellow-700",
  [PillActions.SEEN]: "bg-white border border-brand-gray-100 text-brand-gray-500",
};

const pillIcons: IconMap = {
  [PillActions.ASSIGNED]: () => <ArrowBendDownRight className="text-brand-gray-500" weight="bold" size="14px" />,
  [PillActions.RESOLVED]: () => <CheckCircle weight="fill" className="text-brand-green-700" size="14px" />,
  [PillActions.ALERTED]: () => <Warning weight="fill" className="text-brand-purple-500" size="14px" />,
  [PillActions.HIGHLIGHTED]: () => <CircleWavyCheck weight="fill" className="text-brand-purple-500" size="14px" />,
  [PillActions.HIGH_PRIORITY]: () => <Fire weight="fill" className="text-brand-red-600" size="14px" />,
  [PillActions.MEDIUM_PRIORITY]: () => <Fire weight="fill" className="text-brand-orange-500" size="14px" />,
  [PillActions.LOW_PRIORITY]: () => <Fire weight="fill" className="text-brand-yellow-700" size="14px" />,
  [PillActions.SEEN]: () => <Eye weight="fill" className="text-brand-gray-300" size="14px" />,
};

const pillLabels: StringMap = {
  [PillActions.RESOLVED]: "Resolved",
  [PillActions.HIGH_PRIORITY]: "High",
  [PillActions.MEDIUM_PRIORITY]: "Medium",
  [PillActions.LOW_PRIORITY]: "Low",
  [PillActions.ALERTED]: "Alerted",
  [PillActions.HIGHLIGHTED]: "Highlighted",
  [PillActions.ASSIGNED]: "Assigned",
  [PillActions.SEEN]: "Seen",
};

const priorityToAction: Record<string, PillActions> = {
  high: PillActions.HIGH_PRIORITY,
  medium: PillActions.MEDIUM_PRIORITY,
  low: PillActions.LOW_PRIORITY,
};

export function StatusActionPill(props: { status: string | null; priority: string | null; collapsable?: boolean }) {
  const priority = props.priority?.toLowerCase();
  const status = props.status?.toLowerCase();

  if (status === "resolved") {
    return <ActionPill type={PillActions.RESOLVED} />;
  } else {
    if (!priority) {
      return null;
    }

    return <ActionPill type={priorityToAction[priority]} collapsable={props.collapsable} />;
  }
}
interface ActionPillProps {
  type: PillActions;
  collapsable?: boolean;
  showText?: boolean;
}

export function ActionPill({ type, collapsable, showText = true }: ActionPillProps) {
  const pillClasses = classNames(
    "inline-flex md:mx-1 py-0.5 px-0.5 text-center font-bold rounded-full items-center justify-center align-middle",
    showText && "md:px-2",
    pillStyles[type]
  );

  const icon = pillIcons[type]();
  const labelClasses = classNames(
    "text-xs whitespace-nowrap px-1 font-normal",
    collapsable ? "hidden md:block" : "block"
  );

  return (
    <div className={pillClasses}>
      {showText ? (
        <span>{icon}</span>
      ) : (
        <Tooltip title={pillLabels[type]} placement="top">
          <span>{icon}</span>
        </Tooltip>
      )}
      {showText && <span className={labelClasses}>{pillLabels[type]}</span>}
    </div>
  );
}
