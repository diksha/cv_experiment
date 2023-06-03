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
import React, { useState, useMemo, useRef, useEffect, ReactNode } from "react";
import { DateTime } from "luxon";
import DayPicker, { DateUtils } from "react-day-picker";
import { RiArrowUpSLine, RiArrowDownSLine } from "react-icons/ri";
import { Box, Button } from "@mui/material";
import { ArrowBack } from "@mui/icons-material";
import "react-day-picker/lib/style.css";
import "./DateRangePicker.css";
import { useWindowDimensions } from "shared/hooks";
import { Shortcut, shortcuts } from "./utils";
import { readableDaterange } from "shared/utilities/dateutil";

// react-day-picker's RangeModifier "to" and "from" fields have "Date | undefined | null" typing
export type DateRange = {
  startDate?: DateTime | null;
  endDate?: DateTime | null;
};

interface DatePickerProps {
  onChange: (selectedRange: DateRange) => void;
  onReset?: () => void;
  onBack?: () => void;
  values?: DateRange;
  backValues?: DateRange;
  placeholder?: string;
  wrapperStyle?: string;
  buttonStyle?: string;
  buttonActiveStyle?: string;
  buttonSelectedStyle?: string;
  buttonItemsStyle?: string;
  modalStyle?: string;
  icon?: ReactNode;
  showFooter?: boolean;
  showReset?: boolean;
  overrideBaseStyle?: boolean;
  alignRight?: boolean;
  alignRightMobile?: boolean;
  forceMobileLayout?: boolean;
  noShortcuts?: boolean;
  closeOnSelection?: boolean;
  disabled?: boolean;
  uiKey?: string;
  timezone?: string;
  excludedOptions?: string[];
}

export function DateRangePicker({
  onChange,
  onBack,
  onReset,
  values,
  backValues,
  placeholder = "All Time",
  wrapperStyle,
  buttonStyle = "flex items-center flex-nowrap whitespace-nowrap text-sm px-3 py-1.5 rounded-md border bg-white border-gray-300",
  buttonActiveStyle,
  buttonSelectedStyle,
  buttonItemsStyle = "flex-row-reverse flex gap-2 justify-between items-center",
  icon,
  showFooter = false,
  showReset,
  noShortcuts,
  overrideBaseStyle = true,
  modalStyle = "absolute z-30 shadow-lg bg-white md:w-max min-w-screen md:min-w-auto top-16 rounded-2xl",
  alignRight = true,
  alignRightMobile = true,
  forceMobileLayout = true,
  closeOnSelection = true,
  disabled,
  uiKey,
  timezone: timezoneProp,
  excludedOptions = [],
}: DatePickerProps) {
  const timezone = useMemo(() => timezoneProp || DateTime.now().zoneName, [timezoneProp]);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(false);
  const [showDatePicker, setShowDatePicker] = useState<any>(null);

  const { width } = useWindowDimensions();
  const isMobile = forceMobileLayout || width <= 768;
  const showBackToListButton = isMobile && showDatePicker;

  const { startDate, endDate } = values || { startDate: null, endDate: null };

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

  /**
   * The date picker always returns date objects localized
   * to the browser timezone, so we use setZone to localize
   * these dates to the current site's timezone.
   */
  const localizeDate = (value: Date | DateTime, tz?: string): DateTime => {
    let dateTime;
    if (value instanceof Date) {
      dateTime = DateTime.fromJSDate(value);
    } else {
      dateTime = value;
    }
    return dateTime.setZone(tz || timezone, { keepLocalTime: true });
  };

  const handleClickOutside = (event: MouseEvent) => {
    if (!wrapperRef?.current?.contains(event.target as Node)) {
      setActive(false);
    }
  };

  const handleDayClick = (day: Date) => {
    const { from, to } = DateUtils.addDayToRange(day, {
      from: values?.startDate?.toJSDate(),
      to: values?.endDate?.toJSDate(),
    });
    if (startDate && endDate) {
      const selected = day === from ? from : to;
      onChange({
        startDate: selected && localizeDate(selected).startOf("day"),
        endDate: null,
      });
    } else {
      onChange({
        startDate: from && localizeDate(from).startOf("day"),
        endDate: to && localizeDate(to).endOf("day"),
      });
      if (from && to && closeOnSelection) {
        setActive(false);
      }
    }
  };

  const handleShortcut = (shortcut: Shortcut) => {
    if (shortcut.range) {
      const { startDate, endDate } = shortcut.range(timezone);
      onChange({
        startDate: startDate,
        endDate: endDate,
      });
      setActive(false);
    } else {
      setShowDatePicker(true);
    }
  };

  const handleBackToListClick = () => {
    setShowDatePicker(false);
  };

  const localizedStart = startDate ? localizeDate(startDate, "local") : null;
  const localizedEnd = endDate ? localizeDate(endDate, "local") : null;
  const hasBackValues = backValues?.startDate && backValues?.endDate;

  return (
    <div className={classNames("flex text-base", wrapperStyle)} ref={wrapperRef}>
      {hasBackValues && (
        <button
          data-ui-key={uiKey + "-back-btn"}
          className={buttonStyle}
          style={{ borderRight: "none", borderTopRightRadius: 0, borderBottomRightRadius: 0 }}
          onClick={onBack}
        >
          <ArrowBack sx={{ width: "16px", height: "16px" }} />
        </button>
      )}
      <button
        disabled={disabled}
        data-ui-key={uiKey}
        className={classNames(
          overrideBaseStyle
            ? ""
            : "bg-gray-100 rounded-md border-0 text-gray-500 hover:bg-gray-200 px-4 py-2 min-w-180",
          buttonStyle,
          active ? buttonActiveStyle : "",
          values?.startDate && values?.endDate && !active ? buttonSelectedStyle : "",
          disabled ? "text-brand-gray-200 cursor-not-allowed" : ""
        )}
        style={hasBackValues ? { borderTopLeftRadius: 0, borderBottomLeftRadius: 0 } : {}}
        onClick={() => setActive(!active)}
      >
        <span
          className={classNames(overrideBaseStyle ? "" : "flex gap-4 justify-between items-center", buttonItemsStyle)}
        >
          <div>
            {startDate && endDate ? `${readableDaterange(startDate, endDate)}` : placeholder || "Select date range"}
          </div>
          {icon ? icon : <div>{active ? <RiArrowUpSLine /> : <RiArrowDownSLine />}</div>}
        </span>
      </button>
      {showReset && startDate && endDate && (
        <button
          disabled={disabled}
          className={classNames("text-sm ml-2", disabled ? "text-brand-gray-200 cursor-not-allowed" : "")}
          onClick={onReset}
        >
          Reset
        </button>
      )}
      {active && (
        <div
          className={classNames(
            overrideBaseStyle
              ? modalStyle
              : "absolute z-30 shadow-lg bg-white md:w-max min-w-screen md:min-w-auto rounded-2xl",
            alignRight ? "right-0" : alignRightMobile ? "right-0 md:left-auto md:right-auto" : "left-auto"
          )}
        >
          <div className="flex">
            {!showDatePicker && !noShortcuts && (
              <Shortcuts
                timezone={timezone}
                isMobile={isMobile}
                onChange={handleShortcut}
                excludedOptions={excludedOptions}
              />
            )}
            {(!isMobile || showDatePicker || noShortcuts) && (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                {showBackToListButton ? (
                  <Box sx={{ paddingX: 4, paddingTop: 2, display: { xs: "block", sm: "inline-block" } }}>
                    <Button onClick={handleBackToListClick} variant="outlined" sx={{ width: "100%" }}>
                      Back to List
                    </Button>
                  </Box>
                ) : null}
                <DayPicker
                  className="Selectable"
                  numberOfMonths={2}
                  selectedDays={[
                    localizedStart?.toJSDate()!,
                    { from: localizedStart?.toJSDate(), to: localizedEnd?.toJSDate() },
                  ]}
                  onDayClick={handleDayClick}
                  modifiers={
                    localizedStart && localizedEnd
                      ? { start: localizedStart.toJSDate(), end: localizedEnd.toJSDate() }
                      : undefined
                  }
                />
              </Box>
            )}
          </div>
          {showFooter && (
            <div className="text-right py-2 pr-4 border-t">
              <button onClick={() => setActive(false)} className="border border-brand-blue-900 rounded-sm text-sm px-4">
                Close
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface ShortcutsProps {
  timezone: string;
  isMobile: boolean;
  onChange: (shortcut: Shortcut) => void;
  excludedOptions: string[];
}
function Shortcuts(props: ShortcutsProps) {
  const options = [...shortcuts].filter((shortcut) => !props.excludedOptions.includes(shortcut.label(props.timezone)));
  if (props.isMobile && !props.excludedOptions.includes("Custom range")) {
    options.push({
      label: () => "Custom range",
    });
  }
  return (
    <div className="flex flex-col border-r md:w-auto w-full">
      {options.map((item: Shortcut, idx, arr) => {
        const label = item.label(props.timezone);
        return (
          <div
            key={label}
            className={classNames(
              "py-2 px-6 cursor-pointer text-sm",
              "hover:bg-gray-200",
              idx < arr.length - 1 && "border-b"
            )}
            onClick={() => props.onChange(item)}
          >
            {label}
          </div>
        );
      })}
    </div>
  );
}
