import { DataPanelSection, EventListSkeleton } from "ui";
import { MouseEventHandler, useState } from "react";
import { Typography, Box, useTheme } from "@mui/material";
import { IncidentDetailsDataPanel, IncidentRow } from "features/incidents";
import { LoadingButton } from "@mui/lab";
import { GetDataPanelIncidents_currentUser_site_incidents_edges_node } from "__generated__/GetDataPanelIncidents";

interface IncidentListItem extends GetDataPanelIncidents_currentUser_site_incidents_edges_node {}

interface DataPanelIncidentListProps {
  loading: boolean;
  incidents: IncidentListItem[];
  title: string;
  showLoadMore: boolean;
  handleLoadMore: MouseEventHandler;
  fetchingMore: boolean;
  uiKey: string;
}
export function DataPanelIncidentList({
  loading,
  incidents = [],
  title,
  showLoadMore,
  handleLoadMore,
  fetchingMore,
  uiKey,
}: DataPanelIncidentListProps) {
  const theme = useTheme();

  if (loading) {
    return <EventListSkeleton />;
  }

  return (
    <DataPanelSection title={title} bodyPadding={0}>
      {incidents.length === 0 ? (
        <Box>
          <Typography textAlign="center" color={theme.palette.grey[500]} padding={4}>
            No events detected during this time
          </Typography>
        </Box>
      ) : (
        <>
          {incidents.map((incident) => (
            <ListItem key={incident.id} incident={incident} uiKey={uiKey} />
          ))}
          {showLoadMore ? (
            <Box sx={{ borderTop: "1px solid", borderColor: theme.palette.grey[300] }}>
              <LoadingButton onClick={handleLoadMore} sx={{ width: "100%" }} loading={fetchingMore}>
                Load More
              </LoadingButton>
            </Box>
          ) : null}
        </>
      )}
    </DataPanelSection>
  );
}

interface ListItemProps {
  incident: IncidentListItem;
  uiKey: string;
}
function ListItem({ incident, uiKey }: ListItemProps) {
  const [dataPanelOpen, setDataPanelOpen] = useState(false);

  const handleClick = () => {
    setDataPanelOpen(true);
  };
  const handleClose = () => {
    setDataPanelOpen(false);
  };

  return (
    <>
      <IncidentRow incident={incident} active={false} onClick={handleClick} uiKey={uiKey} mobileOnly />
      <IncidentDetailsDataPanel incidentUuid={incident.uuid} open={dataPanelOpen} onClose={handleClose} />
    </>
  );
}
