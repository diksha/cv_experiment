import { CalendarBlank } from "phosphor-react";
import { StickyHeader, ContentWrapper, DateRangePicker, DateRange } from "ui";
import { DATE_VALUE_FORMAT } from "features/incidents";
import { forwardRef, useCallback, useState } from "react";
import { useFilters } from "./hooks";
import { Filters } from "./Filters";
import { FilterBag, FilterConfig } from "./types";
import classNames from "classnames";

interface FilterBarProps {
  title?: string;
  dateRangeFilter?: DateRange;
  defaultDateRangeFilter?: DateRange;
  setDateRangeFilter: React.Dispatch<React.SetStateAction<DateRange | undefined>>;
  config: FilterConfig;
  showDatePickerResetButton?: boolean;
  sticky?: boolean;
  variant?: "inline" | "fixed";
}

export const FilterBar = forwardRef<HTMLDivElement, FilterBarProps>(
  (
    {
      title,
      dateRangeFilter,
      defaultDateRangeFilter,
      setDateRangeFilter,
      config,
      showDatePickerResetButton = true,
      sticky = true,
      variant = "inline",
    }: FilterBarProps,
    ref
  ) => {
    const { filterBag, setFilters } = useFilters();
    const [headerStuck, setHeaderStuck] = useState(false);

    const handleDateRangeChange = useCallback(
      (range?: DateRange) => {
        setDateRangeFilter(range);
        setFilters((previous: FilterBag) => {
          if (range) {
            return {
              ...previous,
              startDate: {
                value: range.startDate ? range.startDate.toFormat(DATE_VALUE_FORMAT) : null,
              },
              endDate: {
                value: range.endDate ? range.endDate.toFormat(DATE_VALUE_FORMAT) : null,
              },
            };
          }

          // Date range was cleared, remove it from filteres
          const newFilters = { ...previous };
          delete newFilters["startDate"];
          delete newFilters["endDate"];
          return newFilters;
        });
      },
      [setDateRangeFilter, setFilters]
    );

    const handleDateRangeReset = useCallback(() => {
      handleDateRangeChange(defaultDateRangeFilter);
    }, [handleDateRangeChange, defaultDateRangeFilter]);

    const handleClearFilters = useCallback(() => {
      setFilters({});
      handleDateRangeReset();
    }, [setFilters, handleDateRangeReset]);

    const handleStuck = () => {
      setHeaderStuck(true);
    };
    const handleUnstuck = () => {
      setHeaderStuck(false);
    };
    const stickyHeaderClasses = classNames("transition-colors duration-75", {
      "bg-white border-b border-brand-gray-050": headerStuck,
    });

    const barFilters = (
      <div className="bg-white py-4 px-2 rounded-t-md" ref={ref}>
        <div className="flex md:flex-nowrap md:gap-2">
          <div className="flex gap-4 px-4 md:px-0 justify-between content-center items-center">
            {title && (
              <div className="hidden md:block pt-1 pl-2 font-epilogue font-bold text-lg text-brand-gray-500">
                {title}
              </div>
            )}
            <DateRangePicker
              uiKey="button-filter-by-date-range"
              onChange={handleDateRangeChange}
              onReset={handleDateRangeReset}
              values={dateRangeFilter}
              showReset={!!showDatePickerResetButton}
              icon={<CalendarBlank className="h-4 w-4" />}
              alignRight={false}
              alignRightMobile={false}
            />
          </div>
          <div className="w-px bg-gray-300 flex-shrink-0 hidden md:inline-block" />
          <div className="pr-8">
            <Filters
              config={config}
              values={filterBag}
              onChange={setFilters}
              onClear={handleClearFilters}
              variant={variant}
            />
          </div>
        </div>
      </div>
    );

    return sticky ? (
      <StickyHeader
        className={stickyHeaderClasses}
        zIndex={50}
        top={64}
        onStuck={handleStuck}
        onUnStuck={handleUnstuck}
      >
        <ContentWrapper>{barFilters}</ContentWrapper>
      </StickyHeader>
    ) : (
      barFilters
    );
  }
);
