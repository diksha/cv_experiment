import { useTheme, Box, Typography, Paper, BoxProps } from "@mui/material";
import { Table } from "./Table";
import { TableProps } from "./types";
import { GroupOutlined, WarehouseOutlined, SvgIconComponent } from "@mui/icons-material";
import { ReactNode } from "react";

type IconVariant = "group" | "warehouse";

const Icons: { [key in IconVariant]: SvgIconComponent } = {
  group: GroupOutlined,
  warehouse: WarehouseOutlined,
};

type TableCardProps<T, K extends keyof T> = TableProps<T, K> & {
  title: string;
  subtitle?: string;
  boxProps?: BoxProps;
  icon?: IconVariant;
  showHeader?: boolean;
  children?: ReactNode;
};

export function TableCard<T, K extends keyof T>({
  title,
  subtitle,
  boxProps = { sx: { paddingY: 3, paddingX: 2 } },
  icon,
  showHeader = true,
  children,
  ...tableProps
}: TableCardProps<T, K>) {
  const theme = useTheme();

  let Icon;
  if (icon) {
    Icon = Icons[icon];
  }

  return (
    <Paper data-ui-key={tableProps.uiKey}>
      <Box {...boxProps}>
        {showHeader && (
          <Box sx={{ display: "flex", justifyContent: "space-between" }}>
            <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, paddingBottom: 2 }}>
              <Typography variant="h4">{title}</Typography>
              {subtitle ? <Typography sx={{ color: theme.palette.grey[500] }}>{subtitle}</Typography> : null}
            </Box>
            {Icon ? (
              <Box
                display="flex"
                alignItems="center"
                justifyContent="center"
                height={50}
                width={50}
                borderRadius="50%"
                sx={{ backgroundColor: theme.palette.grey[200] }}
              >
                <Icon />
              </Box>
            ) : null}
          </Box>
        )}
        <Box>
          <Table {...tableProps} />
        </Box>
        {children}
      </Box>
    </Paper>
  );
}
