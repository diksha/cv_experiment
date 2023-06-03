import { Box } from "@mui/material";
import { VideocamOffOutlined } from "@mui/icons-material";
import { Tooltip } from "@mui/material";
import { ASPECT_RATIO_16X9_STYLES } from "features/cameras";

interface CameraThumbnailProps {
  alt: string;
  url?: string | null;
}
export function CameraThumbnail({ alt, url }: CameraThumbnailProps) {
  const tooltipTitle = url ? alt : `${alt} (thumbnail not available)`;
  return (
    <Tooltip title={tooltipTitle} placement="bottom" arrow>
      <Box sx={{ position: "relative", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <Box
          sx={{
            ...ASPECT_RATIO_16X9_STYLES,
            borderRadius: 2,
            backgroundColor: (theme) => theme.palette.grey[900],
            backgroundImage: `url(${url})`,
            backgroundSize: "contain",
            backgroundPosition: "center",
            backgroundRepeat: "no-repeat",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        ></Box>

        {!url ? (
          <VideocamOffOutlined
            sx={{ position: "absolute", height: "50px", width: "50px", color: (theme) => theme.palette.grey[700] }}
          />
        ) : null}
      </Box>
    </Tooltip>
  );
}
