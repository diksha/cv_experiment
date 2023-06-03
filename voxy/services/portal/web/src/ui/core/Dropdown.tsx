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
import classNames from "classnames";
import { Fragment, useState } from "react";
import { Listbox, Transition } from "@headlessui/react";
import { CaretDown, CircleNotch, Check } from "phosphor-react";

export interface DropdownOption {
  id: string | number;
  name: string;
}

interface DropdownProps {
  disabled?: boolean;
  loading?: boolean;
  error?: string;
  options: DropdownOption[];
  onSelect?: (option: DropdownOption) => void;
  value?: string | number;
}

export function Dropdown(props: DropdownProps) {
  const [selected, setSelected] = useState<DropdownOption | undefined>(
    props.value ? props.options.find((option) => option.id === props.value) : undefined
  );

  const onChange = (option: DropdownOption) => {
    setSelected(option);
    props.onSelect?.(option);
  };

  return (
    <Listbox value={selected} onChange={onChange} disabled={props.disabled}>
      {({ open }) => (
        <>
          <div className="relative mt-1">
            <Listbox.Button
              className={classNames(
                "relative w-full cursor-default rounded-lg border bg-white py-2 pl-3 pr-10 text-left shadow-sm sm:text-sm",
                props.disabled
                  ? "text-brand-gray-050 hover:cursor-not-allowed"
                  : props.error
                  ? "border-brand-red-500"
                  : "border-brand-gray-050 hover:border-brand-gray-500 hover:ring-brand-gray-500 focus:border-brand-gray-500 focus:outline-none focus:ring-1 focus:ring-brand-gray-500"
              )}
            >
              {props.loading ? (
                <CircleNotch data-testid="core-dropdown-loading-icon" size={20} className="animate-spin" />
              ) : (
                <span className="block truncate">{selected ? selected.name : "Please select"}</span>
              )}
              <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
                <CaretDown
                  className={classNames("h-5 w-5", props.disabled ? "text-brand-gray-050" : "text-brand-gray-300")}
                  size="12"
                  aria-hidden="true"
                />
              </span>
            </Listbox.Button>

            <Transition
              show={open}
              as={Fragment}
              leave="transition ease-in duration-100"
              leaveFrom="opacity-100"
              leaveTo="opacity-0"
            >
              <Listbox.Options className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-md bg-white py-1 text-base shadow-xl ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
                {props.options.map((option) => (
                  <Listbox.Option
                    key={option.id}
                    className={({ active }) =>
                      classNames(
                        active ? "text-brand-gray-500 bg-brand-gray-050" : "text-gray-900",
                        "relative cursor-default select-none py-2 pl-3 pr-9 border-b border-brand-gray-050 last:border-b-0"
                      )
                    }
                    value={option}
                  >
                    {({ selected }) => (
                      <>
                        <span className={classNames(selected ? "font-semibold" : "font-normal", "block truncate")}>
                          {option.name}
                        </span>

                        {selected ? (
                          <span
                            className={classNames(
                              "absolute inset-y-0 right-0 flex items-center pr-4 text-brand-gray-500"
                            )}
                          >
                            <Check className="h-5 w-5" aria-hidden="true" />
                          </span>
                        ) : null}
                      </>
                    )}
                  </Listbox.Option>
                ))}
              </Listbox.Options>
            </Transition>
          </div>
          {props.error && <div className="text-sm text-brand-red-500 mt-2 font-bold">{props.error}</div>}
        </>
      )}
    </Listbox>
  );
}
