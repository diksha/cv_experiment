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

import React, { ReactNode } from "react";
import classNames from "classnames";
import { FilterValue } from ".";

interface FilterButtonProps {
  label: string;
  active?: boolean;
  icon?: ReactNode;
  borderless?: boolean;
  indicator?: number;
  disabled?: boolean;
  onClick?: (value: FilterValue) => void;
}

export function FilterButton(props: FilterButtonProps) {
  return (
    <div className="relative inline-block">
      <button
        className={classNames(
          "flex items-center flex-nowrap whitespace-nowrap text-sm px-3 py-1.5 rounded-md border",
          props.borderless ? "border-white -ml-2" : props.active ? "border-gray-600" : "border-gray-300",
          props.active ? "bg-gray-100 font-semibold" : "bg-white",
          props.disabled ? "text-gray-400 cursor-not-allowed" : ""
        )}
        onClick={() => props.onClick?.({ value: !props.active })}
        disabled={props.disabled}
      >
        {props.icon || null} {props.label}
      </button>
      {props.indicator && props.indicator > 0 ? (
        <div className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full text-xs bg-gray-900 text-white text-center">
          {props.indicator}
        </div>
      ) : null}
    </div>
  );
}
