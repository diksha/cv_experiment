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
import React, { useMemo } from "react";
import { FilterFormSectionDefinition, FilterBag, FilterValue } from ".";

interface FilterFormSectionProps {
  section: FilterFormSectionDefinition;
  values: FilterBag;
  onChange: (values: FilterBag) => void;
}

const isActive = (currentFilterValue: FilterValue | undefined | null, checkboxValue: string) => {
  if (!currentFilterValue || !currentFilterValue.value) {
    return false;
  } else if (typeof currentFilterValue.value === "boolean") {
    return currentFilterValue.value;
  } else if (typeof currentFilterValue.value === "string") {
    return currentFilterValue.value === checkboxValue;
  } else {
    return currentFilterValue.value.indexOf(checkboxValue) > -1;
  }
};

export function FilterFormSection(props: FilterFormSectionProps) {
  const handleChange = (checkboxValue: string) => {
    const newFilterBag = { ...props.values };
    const existingValue = props.values[props.section.key]?.value;
    if (!existingValue) {
      newFilterBag[props.section.key] = { value: [checkboxValue] };
    } else {
      const foundCheckboxValueAt = (existingValue as string[]).indexOf(checkboxValue);
      if (foundCheckboxValueAt > -1) {
        newFilterBag[props.section.key] = { value: (existingValue as string[]).filter((val) => val !== checkboxValue) };
      } else {
        newFilterBag[props.section.key] = { value: [...(existingValue as string[]), ...[checkboxValue]] };
      }
    }
    props.onChange(newFilterBag);
  };

  const groups = useMemo(() => {
    const allGroups = props.section.groups || [];
    if (props.section.options) {
      allGroups.push({ key: "default-option-group", options: props.section.options });
    }
    return allGroups;
  }, [props.section.groups, props.section.options]);

  return (
    <div>
      <div className="text-md font-semibold font-epilogue text-brand-gray-500 pb-2">{props.section.title}</div>
      <div className="flex flex-col gap-y-1">
        {groups?.map((group) => (
          <React.Fragment key={group.key}>
            {group.label ? <div>{group.label}</div> : null}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-x-12 gap-y-1">
              {group.options?.map((option) => (
                <div className="text-sm" key={option.value}>
                  <label className="whitespace-nowrap cursor-pointer">
                    <input
                      type="checkbox"
                      className="rounded-full cursor-pointer"
                      checked={isActive(props.values?.[props.section.key], option.value)}
                      onChange={() => handleChange(option.value)}
                    />
                    <span className="ml-2 text-gray-500">{option.label}</span>
                  </label>
                </div>
              ))}
            </div>
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
