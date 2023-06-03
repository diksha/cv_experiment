import { TableCard, TableColumnDefinition } from "./table";
import { AssigneeStats } from "features/dashboard";
import { useMemo, useState } from "react";
import { IncidentsByAssigneeDataPanel } from "./IncidentsByAssigneeDataPanel";
import { DateTime } from "luxon";

interface RowData {
  userId: string;
  name: string;
  openCount: number;
  resolvedCount: number;
}
interface IncidentsByAssigneeCardProps {
  data?: (AssigneeStats | null)[] | null;
  startDate: DateTime;
  endDate: DateTime;
  timezone: string;
}
export function IncidentsByAssigneeCard({ data, startDate, endDate, timezone }: IncidentsByAssigneeCardProps) {
  const [dataPanelOpen, setDataPanelOpen] = useState(false);
  const [assigneeId, setAssigneeId] = useState("");
  const [assigneeName, setAssigneeName] = useState("");

  const columns: TableColumnDefinition<RowData, keyof RowData>[] = [
    {
      key: "name",
      header: "Assignee",
    },
    {
      key: "openCount",
      header: "Open",
      textAlign: "right",
    },
    {
      key: "resolvedCount",
      header: "Resolved",
      textAlign: "right",
    },
  ];

  const rows: RowData[] = useMemo(() => {
    const results: RowData[] = [];
    (data || []).forEach((item) => {
      if (item?.assignee?.id && item?.assignee?.fullName) {
        results.push({
          userId: item.assignee.id,
          name: item.assignee.fullName,
          openCount: item.openCount || 0,
          resolvedCount: item.resolvedCount || 0,
        });
      }
    });
    return results;
  }, [data]);

  const handleRowClick = (value: RowData) => {
    setAssigneeId(value.userId);
    setAssigneeName(value.name);
    setDataPanelOpen(true);
  };

  const handleDataPanelClose = () => {
    setDataPanelOpen(false);
  };

  return (
    <>
      <TableCard
        title="Incidents by Assignee"
        subtitle="Last 30 days"
        data={rows}
        columns={columns}
        emptyMessage="No assignments during this time"
        onRowClick={handleRowClick}
        uiKey="datapanel-incidents-by-assignee-card"
      />
      <IncidentsByAssigneeDataPanel
        assigneeId={assigneeId}
        assigneeName={assigneeName}
        open={dataPanelOpen}
        startDate={startDate}
        endDate={endDate}
        timezone={timezone}
        onClose={handleDataPanelClose}
      />
    </>
  );
}
