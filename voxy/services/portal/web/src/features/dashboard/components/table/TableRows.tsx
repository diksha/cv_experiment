import { TableColumnDefinition } from "./types";
import { useTheme, Box } from "@mui/material";
import { TableCell } from "./TableCell";

type TableRowsProps<T, K extends keyof T> = {
  data: Array<T>;
  columns: Array<TableColumnDefinition<T, K>>;
  uiKey: string;
  onRowClick?: (value: T) => void;
};
export function TableRows<T, K extends keyof T>({ data, columns, uiKey, onRowClick }: TableRowsProps<T, K>) {
  const theme = useTheme();
  const rows = data.map((row, rowIndex) => {
    const handleClick = () => {
      onRowClick?.(row);
    };

    return (
      <Box
        component="tr"
        key={`row-${rowIndex}`}
        onClick={handleClick}
        sx={{
          // TODO: get borderRadius from theme
          borderRadius: "8px",
          "&:hover": onRowClick && {
            backgroundColor: theme.palette.grey[200],
            cursor: "pointer",
          },
          transition: (theme) =>
            theme.transitions.create("background-color", {
              duration: theme.transitions.duration.shortest,
            }),
        }}
        data-ui-key={`${uiKey}-table-row`}
      >
        {columns.map((column, columnIndex) => {
          const firstColumn = columnIndex === 0;
          const lastColumn = columnIndex === columns.length - 1;

          return (
            <TableCell
              key={`cell-${columnIndex}`}
              firstColumn={firstColumn}
              lastColumn={lastColumn}
              textAlign={column.textAlign}
            >
              {row[column.key] || "-"}
            </TableCell>
          );
        })}
      </Box>
    );
  });

  const spacer = <Box component="tr" sx={{ height: theme.spacing(1) }}></Box>;

  return (
    <tbody>
      {rows.length > 0 ? spacer : null}
      {rows}
    </tbody>
  );
}
