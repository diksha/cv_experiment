import { ElementType } from "react";
import { AccountTab } from "./enums";

export interface TabConfig {
  id: AccountTab;
  label: string;
  path: string;
  icon: ElementType;
}
