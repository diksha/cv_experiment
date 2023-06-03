import { BarChart as ReBarChart, Bar, ResponsiveContainer } from "recharts";
import { useTheme } from "@mui/material";
import { IncidentAggregateGroupByDate } from "features/dashboard";

interface IncidentTrendBarChartSimpleProps {
  data: IncidentAggregateGroupByDate[];
  height?: number;
}

export function IncidentTrendBarChartSimple({ data, height = 40 }: IncidentTrendBarChartSimpleProps) {
  const theme = useTheme();

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ReBarChart data={data}>
        <Bar dataKey="dateCount" fill={theme.palette.primary[900]} radius={[2, 2, 0, 0]} minPointSize={2} />
      </ReBarChart>
    </ResponsiveContainer>
  );
}
