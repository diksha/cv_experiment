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
import { useState, useMemo, useCallback, useEffect } from "react";
import classNames from "classnames";
import { CaretDown, Check } from "phosphor-react";
import { Popover } from "@headlessui/react";
import { FilterValue, FilterSelectionOption } from ".";

interface FilterSelectionProps {
  label: string;
  active?: boolean;
  indicator?: number;
  disabled?: boolean;
  items: FilterSelectionOption[];
  values?: string[];
  onClick?: (value: FilterValue) => void;
  onChange?: (selected: string[]) => void;
}

export function FilterSelection(props: FilterSelectionProps) {
  const [selected, setSelected] = useState<{ [key: string]: boolean }>({});

  const numSelected = useMemo(() => {
    return Object.keys(selected).reduce((acc, key) => {
      acc = acc + (!!selected[key] ? 1 : 0);
      return acc;
    }, 0);
  }, [selected]);

  const handleOnItemClick = useCallback(
    (item: FilterSelectionOption) => {
      const newSelected = Object.assign({}, selected);
      if (newSelected[item.value]) {
        newSelected[item.value] = false;
      } else {
        newSelected[item.value] = true;
      }
      setSelected(newSelected);
      props.onChange?.(Object.keys(newSelected).filter((key) => !!newSelected[key]));
    },
    [props, selected]
  );

  const handleOnAllClick = useCallback(() => {
    setSelected({});
    props.onChange?.([]);
  }, [props]);

  useEffect(() => {
    if (props.values?.length) {
      setSelected(
        props.values.reduce((acc: Record<string, boolean>, value) => {
          acc[value] = true;
          return acc;
        }, {})
      );
    } else {
      setSelected({});
    }
  }, [props.values]);

  return (
    <div className="relative inline-block">
      <Popover>
        {({ open }) => (
          <>
            <Popover.Button
              className={classNames(
                "flex flex-nowrap items-center text-sm px-3 py-1.5 rounded-md border",
                props.active ? "border-gray-600" : "border-gray-300",
                props.active ? "bg-gray-100 font-semibold" : "bg-white",
                props.disabled ? "text-gray-400 cursor-not-allowed" : ""
              )}
              disabled={props.disabled}
            >
              {props.label}{" "}
              <CaretDown className={classNames("w-4 h-4 ml-1 inline-block", open ? "transform rotate-180" : null)} />
            </Popover.Button>
            <Popover.Panel className="absolute left-0 top-10 z-20 border border-gray-300 bg-white rounded-md p-1 drop-shadow-md">
              <button
                onClick={() => handleOnAllClick()}
                className={classNames(
                  "w-full flex text-sm align-middle px-2 py-1 hover:bg-brand-gray-050 rounded-md",
                  numSelected === 0 ? "font-bold" : "text-brand-gray-200"
                )}
              >
                {numSelected === 0 ? (
                  <Check size={16} className="mr-3" />
                ) : (
                  <span className="w-4 h-4 mr-3">&nbsp;</span>
                )}
                <span className="whitespace-nowrap mr-3">{props.label}</span>
              </button>
              {props.items.map((item: FilterSelectionOption) => (
                <button
                  key={item.value}
                  className={classNames(
                    "w-full flex text-sm align-middle px-2 py-1 hover:bg-brand-gray-050 rounded-md",
                    selected[item.value] ? "font-bold" : null
                  )}
                  onClick={() => handleOnItemClick(item)}
                >
                  {selected[item.value] ? (
                    <Check size={16} className="mr-3" />
                  ) : (
                    <span className="w-4 h-4 mr-3">&nbsp;</span>
                  )}
                  <span className="whitespace-nowrap mr-3">{item.label}</span>
                </button>
              ))}
            </Popover.Panel>
            {props.indicator && props.indicator > 0 ? (
              <div className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full text-xs bg-gray-900 text-white text-center">
                {props.indicator}
              </div>
            ) : null}
          </>
        )}
      </Popover>
    </div>
  );
}
