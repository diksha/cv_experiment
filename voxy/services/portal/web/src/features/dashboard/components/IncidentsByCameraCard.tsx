import { DateTime } from "luxon";
import { TableCard, TableColumnDefinition } from "./table";
import { IncidentAggregateGroup } from "features/dashboard";
import { useMemo, useState } from "react";
import { IncidentsByCameraDataPanel } from "./IncidentsByCameraDataPanel";
import { IncidentType } from "features/dashboard";

interface RowData {
  cameraId: string;
  cameraName: string;
  incidentCount: number;
}
interface IncidentsByCameraCardProps {
  data?: IncidentAggregateGroup[] | null;
  incidentTypes: IncidentType[];
  startDate: DateTime;
  endDate: DateTime;
  timezone: string;
}
export function IncidentsByCameraCard({
  data,
  incidentTypes,
  startDate,
  endDate,
  timezone,
}: IncidentsByCameraCardProps) {
  const [dataPanelOpen, setDataPanelOpen] = useState(false);
  const [selectedCameraId, setSelectedCameraId] = useState("");
  const [selectedCameraName, setSelectedCameraName] = useState("");

  const columns: TableColumnDefinition<RowData, keyof RowData>[] = [
    {
      key: "cameraName",
      header: "Camera",
    },
    {
      key: "incidentCount",
      header: "Total",
      textAlign: "right",
    },
  ];

  const rows: RowData[] = useMemo(() => {
    const cameraToRowMap: { [key: string]: RowData } = {};

    const groups = data || [];

    groups.forEach((row) => {
      if (!cameraToRowMap[row.dimensions.camera.id]) {
        // Initialize row
        cameraToRowMap[row.dimensions.camera.id] = {
          cameraId: row.dimensions.camera.id,
          cameraName: row.dimensions.camera.name,
          incidentCount: 0,
        };
      }

      // Increment count
      cameraToRowMap[row.dimensions.camera.id].incidentCount += row.metrics.count;
    });

    return Object.values(cameraToRowMap).sort((a, b) => a.cameraName.localeCompare(b.cameraName));
  }, [data]);

  const handleRowClick = (value: RowData) => {
    setSelectedCameraId(value.cameraId);
    setSelectedCameraName(value.cameraName);
    setDataPanelOpen(true);
  };

  const handleDataPanelClose = () => {
    setDataPanelOpen(false);
  };

  return (
    <>
      <TableCard
        title="Incidents by Camera"
        subtitle="Last 30 days"
        data={rows}
        columns={columns}
        emptyMessage="No incidents during this time"
        onRowClick={handleRowClick}
        uiKey="datapanel-incidents-by-camera-card"
      />
      <IncidentsByCameraDataPanel
        cameraId={selectedCameraId}
        cameraName={selectedCameraName}
        incidentTypes={incidentTypes}
        open={dataPanelOpen}
        startDate={startDate}
        endDate={endDate}
        timezone={timezone}
        onClose={handleDataPanelClose}
      />
    </>
  );
}
