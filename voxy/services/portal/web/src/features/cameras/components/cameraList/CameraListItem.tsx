import { useState } from "react";
import { ASPECT_RATIO_16X9_STYLES, CameraEditModal, CameraThumbnail } from "features/cameras";
import { EditOutlined } from "@mui/icons-material";
import { useTheme, Chip, Box, Skeleton, IconButton, Typography } from "@mui/material";

const ROW_FLEX_DIRECTION_STYLES = {
  xs: "column",
  sm: "row",
};
const THUMBNAIL_COLUMN_WIDTH_STYLES = {
  xs: "100%",
  sm: "200px",
  md: "300px",
};

interface CameraIncidentType {
  id: string;
  key: string;
  name: string;
  backgroundColor: string;
}

export interface CameraListItemProps {
  id: string;
  name: string;
  incidentTypes: CameraIncidentType[];
  thumbnailUrl?: string | null;
  bannedNames?: string[];
}

export function CameraListItem({ id, name, incidentTypes, thumbnailUrl, bannedNames = [] }: CameraListItemProps) {
  const theme = useTheme();
  const [modalOpen, setModalOpen] = useState(false);

  const handleModalOpen = () => {
    setModalOpen(true);
  };

  const handleModalClose = () => {
    setModalOpen(false);
  };

  return (
    <>
      <Box sx={{ display: "flex", flexDirection: ROW_FLEX_DIRECTION_STYLES, gap: 2, padding: 2 }}>
        <Box sx={{ width: THUMBNAIL_COLUMN_WIDTH_STYLES }}>
          <CameraThumbnail alt={name} url={thumbnailUrl} />
        </Box>
        <Box sx={{ flex: "1", display: "flex", flexDirection: "column", gap: 1 }}>
          <Box sx={{ display: "flex", gap: 2, alignItems: "center" }}>
            <Box sx={{ flex: "1" }}>
              <Typography variant="h3">{name}</Typography>
            </Box>
            <IconButton onClick={handleModalOpen}>
              <EditOutlined />
            </IconButton>
          </Box>
          {incidentTypes.length > 0 ? (
            <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
              {incidentTypes.map((incidentType) => (
                <Chip key={incidentType.id} label={incidentType.name} />
              ))}
            </Box>
          ) : (
            <Box sx={{ color: theme.palette.grey[500] }}>No incident types are enabled for this camera</Box>
          )}
        </Box>
      </Box>
      <CameraEditModal
        cameraId={id}
        cameraName={name}
        open={modalOpen}
        bannedNames={bannedNames}
        onClose={handleModalClose}
      />
    </>
  );
}

export function CameraListItemSkeleton() {
  return (
    <Box sx={{ display: "flex", flexDirection: ROW_FLEX_DIRECTION_STYLES, gap: 2, padding: 2 }}>
      <Box sx={{ width: THUMBNAIL_COLUMN_WIDTH_STYLES }}>
        <Skeleton variant="rounded" sx={{ ...ASPECT_RATIO_16X9_STYLES }}></Skeleton>
      </Box>
      <Box sx={{ flex: "1", display: "flex", flexDirection: "column", gap: 2 }}>
        <Box>
          <Skeleton variant="rounded" height="30px" width="60%" />
        </Box>
        <Box></Box>
      </Box>
    </Box>
  );
}
