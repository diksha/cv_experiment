import React from "react";
import { Permission } from "features/auth";
import { SvgIconProps } from "@mui/material";

export type TabConfig = {
  id: string;
  title: string;
  globalPermission?: Permission;
  route: string;
  active: boolean;
  icon?: React.ReactElement<SvgIconProps>;
};
