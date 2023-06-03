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
import { RadioGroup } from "@headlessui/react";
import classNames from "classnames";

export type OptionProps = {
  name: string;
  value: string;
  description?: string;
  icon: React.ReactNode;
  iconClass: string;
  selectedIconClass: string;
  index?: number;
  uiKey?: string;
};

type IconRadioGroupProps = {
  label?: string;
  options: OptionProps[];
  selected?: string;
  onChange: (option: string) => void;
};

export function IconRadioGroup({ label, options, selected, onChange }: IconRadioGroupProps) {
  return (
    <RadioGroup value={selected} onChange={onChange}>
      {label ? <RadioGroup.Label className="sr-only">{label}</RadioGroup.Label> : null}
      <div className="bg-white rounded-md -space-y-px">
        {options.map((option, index) => (
          <RadioGroup.Option
            key={option.name}
            data-ui-key={option.uiKey}
            value={option.value}
            className={({ checked }) =>
              classNames(
                index === 0 ? "rounded-tl-md rounded-tr-md" : "",
                index === options.length - 1 ? "rounded-bl-md rounded-br-md" : "",
                checked ? "bg-blue-50 border-gray-500 z-10" : "border-gray-200",
                "relative border p-4 flex gap-4 justify-center items-center cursor-pointer focus:outline-none"
              )
            }
          >
            {({ active, checked }) => (
              <>
                <span className={classNames("flex-grow-0", checked ? option.selectedIconClass : option.iconClass)}>
                  {option.icon}
                </span>
                <div className="flex-grow">
                  <RadioGroup.Label
                    as="div"
                    className={classNames(
                      checked ? "text-brand-blue-900 font-bold" : "text-gray-900 font-medium",
                      "block text-sm"
                    )}
                  >
                    {option.name}
                  </RadioGroup.Label>
                  <RadioGroup.Description
                    as="div"
                    className={classNames(checked ? "text-gray-700" : "text-gray-500", "block text-sm")}
                  >
                    {option.description}
                  </RadioGroup.Description>
                </div>
              </>
            )}
          </RadioGroup.Option>
        ))}
      </div>
    </RadioGroup>
  );
}
