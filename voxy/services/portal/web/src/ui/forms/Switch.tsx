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
import { Switch as HSwitch } from "@headlessui/react";
import classNames from "classnames";

interface SwitchProps {
  checked: boolean;
  onChange: () => void;
}
export function Switch(props: SwitchProps) {
  return (
    <div className="inline-block">
      <HSwitch
        checked={props.checked}
        onChange={props.onChange}
        className={classNames(
          props.checked ? "bg-brand-green-600" : "bg-gray-200",
          "relative inline-flex flex-shrink-0 h-3 w-6 border-2 border-transparent rounded-full cursor-pointer transition-colors ease-in-out duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:white"
        )}
      >
        <span className="sr-only">Use setting</span>
        <span
          aria-hidden="true"
          className={classNames(
            props.checked ? "translate-x-3" : "translate-x-0",
            "pointer-events-none inline-block h-2 w-2 rounded-full bg-white shadow transform ring-0 transition ease-in-out duration-200"
          )}
        />
      </HSwitch>
    </div>
  );
}
