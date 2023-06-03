import { TableHeader } from "./TableHeader";
import { TableRows } from "./TableRows";
import { TableProps } from "./types";
import { Box } from "@mui/material";

export function Table<T, K extends keyof T>({ data, columns, uiKey, emptyMessage, onRowClick }: TableProps<T, K>) {
  const empty = data.length === 0;
  return (
    <>
      <Box
        component="table"
        sx={{
          borderCollapse: "collapse",
          width: "100%",
        }}
      >
        <TableHeader columns={columns} />
        <TableRows data={data} columns={columns} onRowClick={onRowClick} uiKey={uiKey} />
      </Box>
      {empty && emptyMessage ? (
        <Box
          sx={{
            width: "100%",
            textAlign: "center",
            paddingTop: 3,
            paddingX: 3,
            color: (theme) => theme.palette.grey[500],
          }}
        >
          {emptyMessage}
        </Box>
      ) : null}
    </>
  );
}
