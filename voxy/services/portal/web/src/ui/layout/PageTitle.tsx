import { Box, Typography } from "@mui/material";

interface PageTitleProps {
  title: string;
  secondaryTitle?: string;
  boxPadding?: string;
}

export function PageTitle({ title, secondaryTitle, boxPadding = "32px 0" }: PageTitleProps) {
  return (
    <Box padding={boxPadding}>
      <Typography sx={{ fontSize: "18px", fontWeight: "700" }}>{title}</Typography>
      {secondaryTitle && <Typography sx={{ marginTop: "12px" }}>{secondaryTitle}</Typography>}
    </Box>
  );
}
