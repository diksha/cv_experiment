import { useLazyQuery } from "@apollo/client";
import { Box, Typography, useTheme } from "@mui/material";
import { GET_EXECUTIVE_DASHBOARD_ACTIVE_EMPLOYEES } from "../../queries";
import { DateTime } from "luxon";
import { useEffect, useMemo, useState } from "react";
import { DataPanel, DateRange, DateRangePicker } from "ui";
import { CalendarToday } from "@mui/icons-material";
import { UserSession } from "features/executive-dashboard";
import { filterNullValues } from "shared/utilities/types";
import { MostActiveEmployees } from "../activity";
import {
  GetExecutiveDashboardActiveEmployees,
  GetExecutiveDashboardActiveEmployeesVariables,
} from "__generated__/GetExecutiveDashboardActiveEmployees";

interface MostActiveEmployeesDataPanelProps {
  open: boolean;
  startDate: DateTime;
  endDate: DateTime;
  onClose: () => void;
}
export function MostActiveEmployeesDataPanel({ open, startDate, endDate, onClose }: MostActiveEmployeesDataPanelProps) {
  const theme = useTheme();
  const [dateRangeFilter, setDateRangeFilter] = useState<DateRange>({
    startDate,
    endDate,
  });
  const timezone = DateTime.now().zoneName;

  const [GetExecutiveDashboardActiveEmployees, { data, loading }] = useLazyQuery<
    GetExecutiveDashboardActiveEmployees,
    GetExecutiveDashboardActiveEmployeesVariables
  >(GET_EXECUTIVE_DASHBOARD_ACTIVE_EMPLOYEES, {
    fetchPolicy: "network-only",
    variables: {
      startDate: dateRangeFilter?.startDate?.toISODate(),
      endDate: dateRangeFilter?.endDate?.toISODate(),
    },
  });

  useEffect(() => {
    if (open) {
      setDateRangeFilter({
        startDate,
        endDate,
      });
      GetExecutiveDashboardActiveEmployees();
    }
  }, [open, GetExecutiveDashboardActiveEmployees, startDate, endDate]);

  const handleDateRangeChange = (dateRange: DateRange) => {
    setDateRangeFilter(dateRange);
  };

  const handleClose = () => {
    setDateRangeFilter({
      startDate,
      endDate,
    });
    onClose();
  };

  const table = useMemo(() => {
    const userSessions = filterNullValues<UserSession>(data?.currentUser?.organization?.sessionCount.users);
    return (
      <MostActiveEmployees
        userSessions={userSessions}
        startDate={dateRangeFilter.startDate as DateTime}
        endDate={dateRangeFilter.endDate as DateTime}
        loading={loading}
        tableBoxProps={{ sx: { paddingY: 1 } }}
        clickable
      />
    );
  }, [data, dateRangeFilter.startDate, dateRangeFilter.endDate, loading]);

  return (
    <DataPanel open={open} onClose={handleClose}>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }} data-ui-key="most-active-employees-data-panel">
        <Box sx={{ display: "flex", alignItems: "center" }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h2" sx={{ marginBottom: "8px" }}>
              Most Active Employees
            </Typography>
            <Typography sx={{ color: theme.palette.grey[500] }}>Total Sessions</Typography>
          </Box>
          <Box>
            <DateRangePicker
              uiKey="incident-trend-data-panel-daterange-filter"
              onChange={handleDateRangeChange}
              values={dateRangeFilter}
              modalStyle="absolute z-30 shadow-lg bg-white md:w-max min-w-screen md:min-w-auto top-16 rounded-2xl right-3"
              alignRight={false}
              alignRightMobile={false}
              icon={<CalendarToday sx={{ height: 16, width: 16 }} />}
              timezone={timezone}
              excludedOptions={["Today", "Yesterday", "Custom range"]}
            />
          </Box>
        </Box>
        {table}
      </Box>
    </DataPanel>
  );
}
