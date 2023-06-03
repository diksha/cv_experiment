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

export function Checkbox(props: {
  name: string;
  checked: boolean;
  label: string;
  description?: string;
  disabled?: boolean;
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div className="relative flex items-start ">
      <div className="flex items-center h-5">
        <input
          id={props.name}
          aria-describedby={props.label}
          name={props.label}
          checked={props.checked}
          type="checkbox"
          onChange={props.onChange}
          className="focus:ring-voxel-blue-500 h-4 w-4 border-gray-300 rounded cursor-pointer"
        />
      </div>
      <div className="ml-3 text-sm">
        <label htmlFor={props.name} className="font-bold text-gray-700 cursor-pointer">
          {props.label}
        </label>
        {props.description ? <p className="text-gray-500">{props.description}</p> : null}
      </div>
    </div>
  );
}
