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
import { Spinner } from "ui";
import { useEffect, useState } from "react";
import styles from "./legends.module.css";

export interface LegendItem {
  backgroundColor: string;
  label: string;
  value: string;
}

export const ChartLegend = (props: {
  title: string;
  loading: boolean;
  items: LegendItem[];
  showSelection?: boolean;
  onChange?: (val: string[]) => void;
  selectionKey?: keyof LegendItem;
}) => {
  const [checked, setChecked] = useState<Record<string, boolean>>({});
  useEffect(() => {
    if (props.showSelection && props.selectionKey) {
      const map: Record<string, boolean> = {};
      props.items &&
        props.items.forEach((item: LegendItem) => {
          if (props.selectionKey) {
            map[item[props.selectionKey]] = true;
          }
        });
      setChecked(map);
    }
  }, [props.selectionKey, props.showSelection, props.items]);

  const handleOnChange = (item: LegendItem) => {
    if (props.selectionKey && !(props.selectionKey in item)) {
      return;
    }
    const newChecked: Record<string, boolean> = { ...checked, [item.value]: !checked[item.value] };
    if (props.onChange) {
      const filtered = Object.keys(newChecked).filter((key) => newChecked[key]);
      props.onChange(filtered);
    }
    setChecked(newChecked);
  };

  return (
    <>
      {props.loading ? (
        <div className="grid justify-center p-4 opacity-40">
          <div>
            <Spinner />
          </div>
        </div>
      ) : (
        <div className={styles.legend}>
          <div className="font-medium">{props.title}:</div>
          {Object.keys(checked).length &&
            props.items.map((item: LegendItem, idx) => (
              <label
                key={item.value}
                title="Select to hide/display line"
                className={classNames(
                  "ml-5 md:ml-0 mb-5 md:mb-0",
                  props.showSelection && props.selectionKey
                    ? ["block cursor-pointer md:inline-block", styles.selection]
                    : null
                )}
              >
                <input
                  className={classNames("mr-1.5", { "md:ml-3": idx !== 0 })}
                  type="checkbox"
                  checked={checked[item[props.selectionKey || "value"]]}
                  onChange={() => {
                    if (props.selectionKey && props.onChange) {
                      handleOnChange(item);
                    }
                  }}
                />
                <span className="border-b-3 pb-1.5 font-medium" style={{ borderColor: item.backgroundColor }}>
                  {item.label}
                </span>
              </label>
            ))}
        </div>
      )}
    </>
  );
};
