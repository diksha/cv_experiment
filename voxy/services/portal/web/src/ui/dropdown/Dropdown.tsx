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
import React, { useState, useEffect, useRef } from "react";
import classNames from "classnames";
import { RiArrowUpSLine, RiArrowDownSLine, RiCheckFill } from "react-icons/ri";
import styles from "./Dropdown.module.css";
import { FilterOptions } from "shared/types";

interface DropdownProps {
  loading?: boolean;
  selectedValue?: string;
  selectedValues?: string[];
  options: { value: string; label: string; default?: boolean }[];
  fetchOnChange?: boolean;
  onChange: (newSelectedValue: string, options: FilterOptions) => void;
  onActivate?: (event: React.MouseEvent) => void;
  wrapperStyle?: string;
  buttonStyle?: string;
}

export function Dropdown(props: DropdownProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(false);
  const { wrapperStyle, buttonStyle, selectedValue, options, fetchOnChange, onChange } = props;
  const selectedOption = options.find((x) => x.value === selectedValue);

  useEffect(() => {
    if (active) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      if (active) {
        document.removeEventListener("mousedown", handleClickOutside);
      }
    };
  }, [active]);

  const toggleActive = () => {
    setActive(!active);
  };

  const handleClickOutside = (event: MouseEvent) => {
    if (!wrapperRef?.current?.contains(event.target as Node)) {
      setActive(false);
    }
  };

  const handleSelected = (value: string) => {
    onChange(value, { fetchOnChange });
    setActive(false);
  };

  return (
    <div className={classNames("relative", "text-base", wrapperStyle)} ref={wrapperRef} onClick={props.onActivate}>
      <button
        style={{ minWidth: 180 }}
        className={classNames(
          "bg-gray-100",
          "rounded-md",
          "border-0",
          "text-gray-500",
          "px-4",
          "py-2",
          "hover:bg-gray-200",
          buttonStyle
        )}
        onClick={toggleActive}
      >
        {props.loading ? (
          <div>Loading</div>
        ) : (
          <span className={classNames("flex", "justify-between", "items-center")}>
            <div>{selectedOption?.label}</div>
            <div>{active ? <RiArrowUpSLine /> : <RiArrowDownSLine />}</div>
          </span>
        )}
      </button>
      <div
        className={classNames(styles.options, "shadow-lg", {
          [styles.active]: active,
        })}
      >
        {options.map((option) => (
          <Option
            key={option.value}
            value={option.value}
            label={option.label}
            selectedValue={selectedValue}
            onClick={() => handleSelected(option.value)}
          />
        ))}
      </div>
    </div>
  );
}

function Option(props: any) {
  const { value, label, onClick, selectedValue } = props;
  const active = value === selectedValue;
  return (
    <div className={styles.option} onClick={onClick} data-active={active ? "true" : "false"}>
      <div className={styles.icon}>{active && <RiCheckFill />}</div>
      <div>{label}</div>
    </div>
  );
}
