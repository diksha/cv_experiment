import { useTheme } from "@mui/material";
import { Box, Typography } from "@mui/material";

interface StatisticCardValueProps {
  label?: string;
  value?: number | null;
}
export function StatisticCardValue({ label, value }: StatisticCardValueProps) {
  const theme = useTheme();

  return (
    <Box display="flex" flexDirection="column" gap={0}>
      <Typography fontSize={40} fontWeight={800}>
        {value || 0}
      </Typography>
      {label ? (
        <Typography color={theme.palette.grey[500]} mt={-1}>
          {label}
        </Typography>
      ) : null}
    </Box>
  );
}
