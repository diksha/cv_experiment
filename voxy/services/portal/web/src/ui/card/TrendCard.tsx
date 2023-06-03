import { Box, Paper, PaperProps, Typography, useMediaQuery, useTheme } from "@mui/material";
import { IncidentTrend, IncidentTrendBarChartSimple, PercentageChange } from "features/dashboard";

interface TrendCardProps {
  trend: IncidentTrend;
  title: string;
  secondaryTitle: string;
  paperProps?: PaperProps;
  onClick?: () => void;
}
export function TrendCard({ trend, title, secondaryTitle, paperProps, onClick }: TrendCardProps) {
  const theme = useTheme();
  const xsBreakpoint = useMediaQuery(theme.breakpoints.down("sm"));
  return (
    <>
      <Paper
        sx={{ width: "100%", ":hover": { boxShadow: 1, cursor: "pointer" } }}
        {...paperProps}
        onClick={onClick}
        data-ui-key="incident-trend-data-panel"
      >
        <Box sx={{ padding: 3, display: { xs: "block", sm: "flex" } }}>
          <Box sx={{ width: { xs: "100%", sm: "50%" } }}>
            <Box display="flex" justifyContent="space-between">
              <Typography variant="h4" sx={{ marginBottom: theme.spacing(1) }}>
                {title}
              </Typography>
              {xsBreakpoint && <PercentageChange trend={trend} />}
            </Box>
            <Typography fontSize={40} fontWeight={800} lineHeight={1.12}>
              {trend.countTotal.toLocaleString()}
            </Typography>
            <Typography sx={{ color: theme.palette.grey[500] }}>{secondaryTitle}</Typography>
          </Box>
          <Box sx={{ width: { xs: "100%", sm: "50%" } }}>
            {!xsBreakpoint && (
              <Box
                sx={{
                  display: "flex",
                  justifyContent: "right",
                  alignItems: "center",
                  marginBottom: theme.spacing(3),
                  minHeight: "28px",
                }}
              >
                <PercentageChange trend={trend} />
              </Box>
            )}
            <IncidentTrendBarChartSimple data={trend.mergedOneDayGroups} />
          </Box>
        </Box>
      </Paper>
    </>
  );
}
