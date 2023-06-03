import { useTheme } from "@mui/material";
import { Box, IconButton, Typography, Stack } from "@mui/material";
import { ScoreGauge } from "features/dashboard";
import { default as Grid } from "@mui/material/Unstable_Grid2";
import { ChevronRight } from "@mui/icons-material";

interface ScoreListItem {
  id: string;
  title: string;
  subtitle: string;
  score: number;
}

interface ScoreListProps {
  title: string;
  columnLabel: string;
  items: ScoreListItem[];
}

export function ScoreList({ title, columnLabel, items }: ScoreListProps) {
  const theme = useTheme();
  return (
    <Box>
      <Box sx={{ paddingBottom: "1rem" }}>
        <Typography variant="h4">{title}</Typography>
      </Box>
      <Stack spacing={1}>
        <Grid
          container
          sx={{ fontWeight: "bold", backgroundColor: theme.palette.grey[200], borderRadius: "8px" }}
          spacing={0}
        >
          <Grid xs={7}>{columnLabel}</Grid>
          <Grid xs={2}>Score</Grid>
          <Grid xs={2}>Trend</Grid>
          <Grid xs={1}></Grid>
        </Grid>
        {items.map((item) => (
          <Grid key={item.id} container spacing={0} sx={{ alignItems: "center" }}>
            <Grid xs={7}>
              <Typography sx={{ fontWeight: "bold" }}>{item.title}</Typography>
              <Typography sx={{ color: theme.palette.grey[500] }}>{item.subtitle}</Typography>
            </Grid>
            <Grid xs={2}>
              <ScoreGauge score={item.score} size="small" />
            </Grid>
            <Grid xs={2}>TODO</Grid>
            <Grid xs={1}>
              <IconButton sx={{ backgroundColor: theme.palette.grey[200], borderRadius: "8px" }}>
                <ChevronRight />
              </IconButton>
            </Grid>
          </Grid>
        ))}
      </Stack>
    </Box>
  );
}
