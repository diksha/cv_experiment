import { Fragment } from "react";
import { Box, Card, Divider } from "@mui/material";
import { CameraListItem, CameraListItemProps, CameraListItemSkeleton } from "./CameraListItem";

interface CameraListProps {
  cameras: CameraListItemProps[];
  loading: boolean;
}

export function CameraList({ cameras, loading }: CameraListProps) {
  const bannedNames = cameras.map((node) => node.name);

  if (loading) {
    return <CameraListSkeleton />;
  }

  return (
    <Card>
      {cameras.map((camera, index) => (
        <Fragment key={camera.id}>
          <CameraListItem {...camera} bannedNames={bannedNames} />
          {index !== cameras.length - 1 && <ListItemDivider />}
        </Fragment>
      ))}
    </Card>
  );
}

function CameraListSkeleton() {
  return (
    <Card>
      <Box sx={{ display: "flex", flexDirection: "column" }}>
        <CameraListItemSkeleton />
        <ListItemDivider />
        <CameraListItemSkeleton />
        <ListItemDivider />
        <CameraListItemSkeleton />
      </Box>
    </Card>
  );
}

function ListItemDivider() {
  return <Divider sx={{ borderColor: (theme) => theme.palette.grey[300] }} />;
}
