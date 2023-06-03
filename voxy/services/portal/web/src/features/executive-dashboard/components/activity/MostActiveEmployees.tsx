import { TableCard, TableColumnDefinition } from "features/dashboard";
import { useMemo, useState } from "react";
import { DateTime } from "luxon";
import { readableDaterange } from "shared/utilities/dateutil";
import { UserSession } from "../../types";
import { BoxProps, Button, useTheme } from "@mui/material";
import { UserInfoModal } from "./UserInfoModal";
import { MostActiveEmployeesDataPanel } from "../dataPanels";
import { TableSkeleton } from "ui";

interface RowData {
  userFullName: string;
  value: number;
  session: UserSession;
}
interface MostActiveEmployeesProps {
  userSessions: UserSession[];
  startDate: DateTime;
  endDate: DateTime;
  loading: boolean;
  clickable?: boolean;
  tableBoxProps?: BoxProps;
  showHeader?: boolean;
  showIcon?: boolean;
  showPagination?: boolean;
}
export function MostActiveEmployees({
  userSessions,
  startDate,
  endDate,
  loading,
  clickable,
  tableBoxProps,
  showIcon,
  showHeader,
  showPagination,
}: MostActiveEmployeesProps) {
  const theme = useTheme();
  const [modalOpen, setModalOpen] = useState(false);
  const [modalCurrentUserSession, setModalCurrentUserSession] = useState<UserSession>();
  const [dataPanelOpen, setDataPanelOpen] = useState(false);
  const dateText = readableDaterange(startDate, endDate, "to") || "";

  const columns: TableColumnDefinition<RowData, keyof RowData>[] = [
    {
      key: "userFullName",
      header: "Employee",
    },
    {
      key: "value",
      header: "Total",
      textAlign: "right",
    },
  ];

  const rows: RowData[] = useMemo(() => {
    const results: RowData[] = [];
    const sessions = showPagination ? userSessions.slice(0, 10) : userSessions;
    sessions.forEach((item) => {
      results.push({
        userFullName: item.user?.fullName || "",
        value: item.value,
        session: item,
      });
    });
    return results;
  }, [userSessions, showPagination]);

  const onClick = (row: RowData) => {
    setModalOpen(true);
    setModalCurrentUserSession(row.session);
  };

  const handeModalClose = () => {
    setModalOpen(false);
    setModalCurrentUserSession(undefined);
  };

  const handleDataPanelOpen = () => {
    setDataPanelOpen(true);
  };

  const handleDataPanelClose = () => {
    setDataPanelOpen(false);
  };

  if (loading) {
    return <TableSkeleton />;
  }

  return (
    <>
      <TableCard
        title="Most Active Employees"
        subtitle={`${dateText} - Total Sessions`}
        data={rows}
        columns={columns}
        emptyMessage="No sessions during this time"
        uiKey="most-active-employees-card"
        boxProps={tableBoxProps}
        showHeader={showHeader}
        {...(showIcon ? { icon: "group" } : {})}
        {...(clickable ? { onRowClick: onClick } : {})}
      >
        {showPagination && (
          <Button
            variant="outlined"
            sx={{
              width: "100%",
              marginTop: "12px",
              padding: "8px 0",
              borderRadius: "4px",
              border: `1px solid ${theme.palette.grey[300]}`,
              fontWeight: "400",
            }}
            data-ui-key="most-active-employees-card-see-more-btn"
            onClick={handleDataPanelOpen}
          >
            See All
          </Button>
        )}
      </TableCard>
      {!!modalCurrentUserSession && (
        <UserInfoModal open={modalOpen} onClose={handeModalClose} userSession={modalCurrentUserSession} />
      )}
      <MostActiveEmployeesDataPanel
        open={dataPanelOpen}
        startDate={startDate}
        endDate={endDate}
        onClose={handleDataPanelClose}
      />
    </>
  );
}
