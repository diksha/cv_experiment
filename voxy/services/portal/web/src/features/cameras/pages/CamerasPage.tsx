import { Helmet } from "react-helmet-async";
import { PageTitle } from "ui";
import { useEffect, useMemo } from "react";
import { GET_SITE_CAMERAS, CameraList } from "features/cameras";
import { useLazyQuery } from "@apollo/client";
import { getNodes } from "graphql/utils";
import {
  GetSiteCameras,
  GetSiteCamerasVariables,
  GetSiteCameras_zone_cameras_edges_node,
} from "__generated__/GetSiteCameras";
import { useCurrentUser } from "features/auth";
import { Container, Typography, useMediaQuery } from "@mui/material";

export function CamerasPage() {
  const { currentUser } = useCurrentUser();
  const { site } = currentUser || {};
  const mobileBreakpoint = useMediaQuery("(min-width:768px)");
  const [getSiteCameras, { loading, data }] = useLazyQuery<GetSiteCameras, GetSiteCamerasVariables>(GET_SITE_CAMERAS);

  useEffect(() => {
    if (site) {
      getSiteCameras({
        variables: {
          zoneId: site.id,
        },
      });
    }
  }, [site, getSiteCameras]);

  const cameras = useMemo(() => {
    const nodes = getNodes<GetSiteCameras_zone_cameras_edges_node>(data?.zone?.cameras);
    // Sort the camera by name ASC order
    return nodes.sort((a, b) => (a.name > b.name ? 1 : -1));
  }, [data?.zone?.cameras]);

  return (
    <>
      <Helmet>
        <title>Cameras - Voxel</title>
      </Helmet>
      <Container maxWidth="xl" disableGutters sx={{ padding: 2, margin: 0 }}>
        {mobileBreakpoint ? (
          <PageTitle
            title="Cameras"
            secondaryTitle="Edit camera names for easier incident browsing"
            boxPadding="0 0 30px"
          />
        ) : (
          <Typography sx={{ fontSize: "18px", fontWeight: "700", padding: "18px 12px" }}>Cameras</Typography>
        )}
        <CameraList cameras={cameras} loading={loading} />
      </Container>
    </>
  );
}
