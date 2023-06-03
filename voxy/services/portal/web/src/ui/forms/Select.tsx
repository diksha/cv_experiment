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
import { Select, FormControl, MenuItem, Checkbox, ListItemText, FormControlProps, SelectProps } from "@mui/material";

type MultiSelectProps = {
  FormControlProps?: FormControlProps;
  SelectProps: SelectProps<(string | number)[]>;
  items: any[];
  valueKey: string;
  labelKey: string;
};

export function MultiSelect({ FormControlProps, SelectProps, items, valueKey, labelKey }: MultiSelectProps) {
  return (
    <FormControl size="small" fullWidth {...FormControlProps}>
      <Select multiple displayEmpty MenuProps={{ classes: { paper: "max-h-40" } }} {...SelectProps}>
        {items.map((item) => (
          <MenuItem key={item[valueKey]} value={item[valueKey]}>
            <Checkbox checked={SelectProps.value?.includes(item[valueKey])} classes={{ root: "py-0 pl-0 pr-2" }} />
            <ListItemText primary={item[labelKey]} />
          </MenuItem>
        ))}
      </Select>
      {Array.isArray(SelectProps.value) && SelectProps.value?.length > 0 && (
        <div className="pt-1 text-sm">
          <span>Selected: </span>
          <strong>
            {Array.isArray(SelectProps.value) &&
              SelectProps.value?.map((item) => items.find((option) => option[valueKey] === item)[labelKey]).join(", ")}
          </strong>
        </div>
      )}
    </FormControl>
  );
}
