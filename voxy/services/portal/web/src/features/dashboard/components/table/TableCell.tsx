import { ReactNode } from "react";
import { Box, useTheme } from "@mui/material";

type TableCellProps = {
  children: ReactNode;
  firstColumn: boolean;
  lastColumn: boolean;
  textAlign?: "left" | "center" | "right";
};
export function TableCell({ children, textAlign, firstColumn, lastColumn }: TableCellProps) {
  const theme = useTheme();

  return (
    <Box
      component="td"
      sx={{
        paddingTop: theme.spacing(1),
        paddingBottom: theme.spacing(1),
        textAlign: textAlign || "left",
        borderRadius: firstColumn ? "8px 0 0 8px" : 0,
        borderTopLeftRadius: firstColumn ? "8px" : "",
        borderBottomLeftRadius: firstColumn ? "8px" : "",
        borderToprightRadius: lastColumn ? "8px" : "",
        borderBottomRightRadius: lastColumn ? "8px" : "",
        borderTopRightRadius: lastColumn ? "8px" : "",
        paddingLeft: firstColumn ? theme.spacing(0.5) : 0,
        paddingRight: lastColumn ? theme.spacing(0.5) : theme.spacing(1),
      }}
    >
      {children}
    </Box>
  );
}
