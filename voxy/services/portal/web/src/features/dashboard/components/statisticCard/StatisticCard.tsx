import { ReactNode } from "react";
import { Link } from "react-router-dom";
import { useTheme, Button, Box, Typography, Paper } from "@mui/material";
import { StatisticCardSkeleton } from "./StatisticCardSkeleton";
import { SubdirectoryArrowLeft, SubdirectoryArrowRight, BookmarkBorder, SvgIconComponent } from "@mui/icons-material";

type IconVariant = "arrowRight" | "arrowLeft" | "bookmark";

const Icons: { [key in IconVariant]: SvgIconComponent } = {
  arrowLeft: SubdirectoryArrowLeft,
  arrowRight: SubdirectoryArrowRight,
  bookmark: BookmarkBorder,
};

interface StatisticCardProps {
  title: string;
  loading: boolean;
  children: ReactNode;
  uiKey: string;
  iconVariant?: IconVariant;
  to?: string;
}
export function StatisticCard({ title, loading, children, uiKey, iconVariant, to }: StatisticCardProps) {
  const theme = useTheme();

  if (loading) {
    return <StatisticCardSkeleton />;
  }

  let Icon;
  if (iconVariant) {
    Icon = Icons[iconVariant];
  }

  const content = (
    <Paper sx={{ width: "100%", height: "100%", ":hover": { boxShadow: 1 } }} data-ui-key={uiKey}>
      <Box display="flex" alignItems="center" height="100%" gap={2} p={2}>
        <Box flex="1" alignSelf="start">
          <Typography variant="h4">{title}</Typography>
          <Box display="flex" gap={{ xs: 2, lg: 4 }}>
            {children}
          </Box>
        </Box>
        {Icon ? (
          <Box>
            <Box
              display="flex"
              alignItems="center"
              justifyContent="center"
              height={50}
              width={50}
              borderRadius="50%"
              sx={{ backgroundColor: theme.palette.grey[200] }}
            >
              <Icon />
            </Box>
          </Box>
        ) : null}
      </Box>
    </Paper>
  );

  return to ? (
    <Button to={to} component={Link} sx={{ width: "100%", textAlign: "left", padding: 0 }}>
      {content}
    </Button>
  ) : (
    content
  );
}
