// trunk-ignore(eslint/@typescript-eslint/no-unused-vars): allow unused vars in type overrides
import * as Alert from "@mui/material/Alert";

declare module "@mui/material/Alert" {
  interface AlertPropsColorOverrides {
    primary;
    secondary;
  }
}
