import { BarChart, Bar, ResponsiveContainer } from "recharts";
import { ProductionLineStatusDataPoint } from "./types";
import { useTheme } from "@mui/material";

export interface DowntimeChartProps {
  data: ProductionLineStatusDataPoint[];
}

export function DowntimeChartSimple({ data }: DowntimeChartProps) {
  const theme = useTheme();

  if (data && data.length > 0) {
    return (
      <ResponsiveContainer width="100%" height={80}>
        <BarChart height={80} data={data}>
          <Bar stackId="a" minPointSize={2} dataKey="metrics.uptimeDurationSeconds" fill={theme.palette.primary[900]} />
          <Bar stackId="a" dataKey="metrics.downtimeDurationSeconds" fill="#C9CDCF" />
          <Bar stackId="a" dataKey="metrics.unknownDurationSeconds" fill="#eeeeee" />
        </BarChart>
      </ResponsiveContainer>
    );
  }
  return (
    <ResponsiveContainer width="100%" height={80}>
      <BarChart height={80} data={Array(24).fill(1)}>
        <Bar stackId="a" dataKey={(v) => v} fill="#eeeeee" />
      </BarChart>
    </ResponsiveContainer>
  );
}
