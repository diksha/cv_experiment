import { useState, useMemo, useEffect, ReactNode, useCallback } from "react";
import { IncidentType } from "features/analytics";
import { Card } from "ui";
import { PriorityOption, LineChart } from "./LineChart";
import { LegendItem, ChartLegend } from "./legends";
import { GetAnalyticsPageData_analytics_series } from "__generated__/GetAnalyticsPageData";
import {
  Select,
  useTheme,
  FormControl,
  InputLabel,
  MenuItem,
  SelectChangeEvent,
  Typography,
  Box,
  useMediaQuery,
} from "@mui/material";

interface SeriesDataPoint extends GetAnalyticsPageData_analytics_series {}

const priorityOptions: PriorityOption[] = [
  {
    label: "High Priority",
    backgroundColor: "#c7522d",
    value: "highPriorityCount",
  },
  {
    label: "Medium Priority",
    backgroundColor: "#ed955b",
    value: "mediumPriorityCount",
  },
  {
    label: "Low Priority",
    backgroundColor: "#ffb946",
    value: "lowPriorityCount",
  },
];

interface IncidentChartProps {
  loading: boolean;
  series: SeriesDataPoint[];
  incidentTypes: IncidentType[];
  children?: ReactNode;
}

export function IncidentChart({ loading, series, incidentTypes, children }: IncidentChartProps) {
  const theme = useTheme();
  const [chartType, setChartType] = useState<string>("incidentType");
  const [clonedTypes, setClonedTypes] = useState<IncidentType[]>([]);
  const [clonedPriorities, setClonedPriorities] = useState<PriorityOption[]>(priorityOptions);
  const [width, setWidth] = useState(0);
  const smBreakpoint = useMediaQuery(theme.breakpoints.up("sm"));

  const handleChartTypeChange = (event: SelectChangeEvent) => {
    setChartType(event.target.value);
  };

  const handleIncidentTypeSelection = useMemo(
    () => (selectedValues: string[]) => {
      const selectedTypes = [...incidentTypes].filter((incidentType: IncidentType) =>
        selectedValues.includes(incidentType.key)
      );
      setClonedTypes(selectedTypes);
    },
    [incidentTypes]
  );

  useEffect(() => {
    handleIncidentTypeSelection(incidentTypes.map((type: IncidentType) => type.key));
  }, [handleIncidentTypeSelection, incidentTypes]);

  const handlePrioritySelection = (selectedValues: string[]) => {
    const selectedPriorities = [...priorityOptions].filter((priorityOption: PriorityOption) =>
      selectedValues.includes(priorityOption.value)
    );
    setClonedPriorities(selectedPriorities);
  };

  const incidentTypeOptions: LegendItem[] = useMemo(() => {
    return incidentTypes.map((incidentType: IncidentType) => ({
      value: incidentType.key,
      label: incidentType.name,
      backgroundColor: incidentType.backgroundColor || theme.palette.primary.main,
    }));
  }, [incidentTypes, theme.palette.primary.main]);

  const chartDiv = useCallback((node) => {
    if (node !== null) {
      setWidth(node.getBoundingClientRect().width);
    }
  }, []);

  return (
    <div className="pb-8">
      <Card loading={loading} noPadding>
        <Typography sx={{ padding: "24px 24px 12px", fontSize: "18px", fontWeight: "700" }}>Total Incidents</Typography>
        <Box
          sx={{
            padding: smBreakpoint ? "0 24px 0 8px" : 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
          }}
        >
          {/* TODO(hq): will remove this in the next iteration of topnav */}
          {children}
          <FormControl size="small" sx={{ marginLeft: "24px" }}>
            <InputLabel id="chart-type-select-label">Chart Type</InputLabel>
            <Select
              labelId="chart-type-select-label"
              id="chart-type-select"
              data-ui-key="chart-type-select"
              autoWidth
              value={chartType}
              label="Chart Type"
              onChange={handleChartTypeChange}
              sx={{ height: "34px" }}
            >
              <MenuItem value="incidentType">Incident Types</MenuItem>
              <MenuItem value="incidentPriority">Priorities</MenuItem>
            </Select>
          </FormControl>
        </Box>
        <div style={{ padding: "20px 20px 20px 5px" }} ref={chartDiv}>
          <LineChart
            chartType={chartType}
            data={series}
            height={400}
            width={width}
            incidentTypes={clonedTypes}
            priorities={clonedPriorities}
          />
          {chartType === "incidentType" && (
            <ChartLegend
              title="Incident Types"
              items={incidentTypeOptions}
              loading={loading}
              showSelection={true}
              selectionKey="value"
              onChange={handleIncidentTypeSelection}
            />
          )}
          {chartType === "incidentPriority" && (
            <ChartLegend
              title="Priorities"
              loading={false}
              items={priorityOptions}
              showSelection={true}
              selectionKey="value"
              onChange={handlePrioritySelection}
            />
          )}
        </div>
      </Card>
    </div>
  );
}
