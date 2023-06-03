import { TableColumnDefinition } from "./types";
import { Box, useTheme } from "@mui/material";

type TableHeaderProps<T, K extends keyof T> = {
  columns: Array<TableColumnDefinition<T, K>>;
};

export function TableHeader<T, K extends keyof T>({ columns }: TableHeaderProps<T, K>) {
  const theme = useTheme();

  const headers = columns.map((column, index) => {
    const firstColumn = index === 0;
    const lastColumn = index === columns.length - 1;
    return (
      <Box
        component="th"
        key={`headCell-${index}`}
        sx={{
          color: theme.palette.grey[500],
          borderBottom: "1px solid",
          borderColor: theme.palette.grey[300],
          textAlign: column.textAlign || "left",
          paddingLeft: firstColumn ? theme.spacing(0.5) : 0,
          paddingRight: lastColumn ? theme.spacing(0.5) : theme.spacing(1),
        }}
      >
        {column.header}
      </Box>
    );
  });

  return (
    <Box component="thead">
      <tr>{headers}</tr>
    </Box>
  );
}
