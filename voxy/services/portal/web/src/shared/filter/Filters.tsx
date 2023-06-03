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
import { useMemo, useCallback, useState, useEffect } from "react";
import classNames from "classnames";
import { useQuery } from "@apollo/client";
import { Modal, Slideover } from "ui";
import {
  FilterButton,
  FilterBag,
  FilterSelection,
  FilterForm,
  isFilterValueEmpty,
  FILTER_KEY_ASSIGNMENT,
  FILTER_KEY_INCIDENT_TYPE,
  FILTER_VALUE_ASSIGNED_TO_ME,
  FilterSelectionOption,
  FILTER_VALUE_ASSIGNED_BY_ME,
  FILTER_KEY_CAMERA,
  FILTER_KEY_EXTRAS,
  FILTER_VALUE_BOOKMARKED,
  FILTER_VALUE_HIGHLIGHTED,
  FilterConfig,
  getActiveFilterNum,
} from ".";
import { GET_FILTER_BAR_DATA } from "./queries";
import { Faders, X } from "phosphor-react";
import { GetFilterBarData } from "__generated__/GetFilterBarData";

export interface FiltersProps {
  config: FilterConfig;
  values: FilterBag;
  variant: "inline" | "fixed";
  onChange: (values: FilterBag) => void;
  onClear: () => void;
}

export function Filters(props: FiltersProps) {
  const { onChange, onClear, values, config } = props;
  const [numActiveFilters, setNumActiveFilters] = useState<number>(0);
  const [isModalOpened, setIsModalOpened] = useState<boolean>(false);
  const [isSliderOpened, setIsSliderOpened] = useState<boolean>(false);
  const [currentValues, setCurrentValues] = useState<FilterBag>(values);
  const [formValues, setFormValues] = useState<FilterBag>(values);
  const { data } = useQuery<GetFilterBarData>(GET_FILTER_BAR_DATA);

  const incidentTypes = useMemo(() => {
    const filtered = (data?.currentUser?.organization?.incidentTypes || []).filter((incidentType) => !!incidentType);
    return filtered.map((incidentType) => ({
      label: incidentType?.name,
      value: incidentType?.key,
    })) as FilterSelectionOption[];
  }, [data]);

  const cameras = useMemo(() => {
    const filtered = (data?.currentUser?.site?.cameras?.edges || []).filter((camera) => !!camera);
    return filtered.map((camera) => ({
      label: camera?.node?.name,
      value: camera?.node?.id,
    })) as FilterSelectionOption[];
  }, [data]);

  const assigned = [
    {
      label: "Assigned To Me",
      value: FILTER_VALUE_ASSIGNED_TO_ME,
    },
    {
      label: "Assigned By Me",
      value: FILTER_VALUE_ASSIGNED_BY_ME,
    },
  ];

  const applyFilterChanges = useCallback(
    (newFilterBag: FilterBag) => {
      // Remove fields with empty values
      for (const [key, value] of Object.entries(newFilterBag)) {
        if (isFilterValueEmpty(value)) {
          delete newFilterBag[key];
        }
      }

      setCurrentValues(newFilterBag);
      setNumActiveFilters(getActiveFilterNum(newFilterBag));
      onChange(newFilterBag);
    },
    [onChange, setCurrentValues, setNumActiveFilters]
  );

  const isQuickFilterActive = (sectionKey: string, value: string) => {
    if (sectionKey in currentValues) {
      const filterValue = currentValues[sectionKey]?.value;
      if (Array.isArray(filterValue)) {
        return filterValue.indexOf(value) > -1;
      }
    }
    return false;
  };

  const handleQuickFilterClick = (sectionKey: string, value: string) => {
    let newFilterBag = JSON.parse(JSON.stringify(currentValues));
    if (!newFilterBag?.[sectionKey]?.value) {
      newFilterBag[sectionKey] = { value: [value] };
    } else {
      const foundIdx = (newFilterBag[sectionKey].value as string[]).indexOf(value);
      if (foundIdx > -1) {
        newFilterBag[sectionKey].value = (newFilterBag[sectionKey].value as string[]).filter((val) => val !== value);
      } else {
        newFilterBag[sectionKey].value = [...newFilterBag[sectionKey].value, ...[value]];
      }
    }
    applyFilterChanges(newFilterBag);
  };

  const handleSelectionFilterChange = (sectionKey: string, values: string[]) => {
    let newFilterBag = JSON.parse(JSON.stringify(currentValues));
    newFilterBag[sectionKey] = { value: values };
    applyFilterChanges(newFilterBag);
  };

  const handleOpenAllFiltersModal = () => {
    setFormValues(JSON.parse(JSON.stringify(currentValues || {})));
    setIsModalOpened(true);
  };

  const handleFormApply = () => {
    applyFilterChanges({ ...formValues });
    setIsModalOpened(false);
    setIsSliderOpened(false);
  };

  useEffect(() => {
    applyFilterChanges(values);
  }, [values, applyFilterChanges]);

  return (
    <>
      <div className="hidden md:flex flex-wrap justify-start gap-2">
        {config.allFilters && (
          <FilterButton
            indicator={numActiveFilters}
            active={numActiveFilters > 0}
            label="All Filters"
            onClick={handleOpenAllFiltersModal}
            icon={<Faders size={16} className="mr-1 inline-block" />}
          />
        )}
        {config.incidentTypes && (
          <FilterSelection
            label="Incident Types"
            items={incidentTypes}
            values={currentValues[FILTER_KEY_INCIDENT_TYPE]?.value as string[]}
            active={(currentValues[FILTER_KEY_INCIDENT_TYPE]?.value as string[])?.length > 0}
            indicator={(currentValues[FILTER_KEY_INCIDENT_TYPE]?.value as string[])?.length}
            onChange={(values) => handleSelectionFilterChange(FILTER_KEY_INCIDENT_TYPE, values)}
          />
        )}
        {config.locations && (
          <FilterSelection
            label="Locations"
            items={cameras}
            values={currentValues[FILTER_KEY_CAMERA]?.value as string[]}
            active={(currentValues[FILTER_KEY_CAMERA]?.value as string[])?.length > 0}
            indicator={(currentValues[FILTER_KEY_CAMERA]?.value as string[])?.length}
            onChange={(values) => handleSelectionFilterChange(FILTER_KEY_CAMERA, values)}
          />
        )}
        {config.assigned && (
          <FilterSelection
            label="Assigned"
            items={assigned}
            values={currentValues[FILTER_KEY_ASSIGNMENT]?.value as string[]}
            active={(currentValues[FILTER_KEY_ASSIGNMENT]?.value as string[])?.length > 0}
            indicator={(currentValues[FILTER_KEY_ASSIGNMENT]?.value as string[])?.length}
            onChange={(values) => handleSelectionFilterChange(FILTER_KEY_ASSIGNMENT, values)}
          />
        )}
        {config.bookmarked && (
          <FilterButton
            active={isQuickFilterActive(FILTER_KEY_EXTRAS, FILTER_VALUE_BOOKMARKED)}
            onClick={() => handleQuickFilterClick(FILTER_KEY_EXTRAS, FILTER_VALUE_BOOKMARKED)}
            label="Bookmarked"
          />
        )}
        {config.highlighted && (
          <FilterButton
            active={isQuickFilterActive(FILTER_KEY_EXTRAS, FILTER_VALUE_HIGHLIGHTED)}
            onClick={() => handleQuickFilterClick(FILTER_KEY_EXTRAS, FILTER_VALUE_HIGHLIGHTED)}
            label="Highlighted"
          />
        )}
        {config.clearAll && (
          <FilterButton
            label="Clear All"
            onClick={() => onClear()}
            borderless={numActiveFilters === 0}
            active={numActiveFilters > 0}
            disabled={numActiveFilters === 0}
          />
        )}

        <Modal open={isModalOpened} onClose={() => setIsModalOpened(false)} fitContent={true}>
          <div className="relative pt-4 px-4 pb-4 border-b border-gray-300">
            <div className="text-xl font-bold text-brand-gray-500 font-epilogue">Filters</div>
            <div className="absolute top-5 right-4">
              <X size={18} className="cursor-pointer" onClick={() => setIsModalOpened(false)} />
            </div>
          </div>
          <div className="px-4 pt-4">
            <FilterForm onChange={(values) => setFormValues(values)} values={formValues} config={config} />
          </div>
          <div className="border-t border-gray-300 pt-4 pb-4 px-4 grid grid-cols-2 gap-3">
            <button
              className="block border border-gray-400 rounded-md text-sm text-gray-400 text-center py-2"
              onClick={() => setFormValues({})}
            >
              Clear All
            </button>
            <button
              className="block border border-gray-900 rounded-md text-sm text-white text-center py-2 bg-gray-900"
              onClick={handleFormApply}
            >
              Done
            </button>
          </div>
        </Modal>
      </div>
      <div className="inline md:hidden">
        {props.variant === "fixed" && (
          <button className="fixed bottom-16 right-2 z-30">
            <div
              className={classNames(
                "flex gap-2 py-2 px-4 items-center shadow-lg",
                "rounded-full bg-brand-blue-900 text-white"
              )}
              onClick={() => setIsSliderOpened(true)}
            >
              <Faders size={16} />
              <p>Filters</p>
            </div>
          </button>
        )}
        {props.variant === "inline" && (
          <button
            className={classNames(
              "flex items-center flex-nowrap whitespace-nowrap text-sm px-3 py-1.5 rounded-md border",
              numActiveFilters > 0 ? "border-gray-600 bg-gray-100 font-semibold" : "border-gray-300 bg-white"
            )}
            onClick={() => setIsSliderOpened(true)}
          >
            All Filters
          </button>
        )}
        <Slideover title="Filters" open={isSliderOpened} onClose={() => setIsSliderOpened(false)}>
          <FilterForm onChange={(values) => setFormValues(values)} values={formValues} config={config} />
          <div className="border-t border-gray-300 pt-4 pb-4 px-4 grid grid-cols-2 gap-3">
            <button
              className="block border border-gray-400 rounded-md text-sm text-gray-400 text-center py-2"
              onClick={() => setFormValues({})}
            >
              Clear All
            </button>
            <button
              className="block border border-gray-900 rounded-md text-sm text-white text-center py-2 bg-gray-900"
              onClick={handleFormApply}
            >
              Done
            </button>
          </div>
        </Slideover>
      </div>
    </>
  );
}
