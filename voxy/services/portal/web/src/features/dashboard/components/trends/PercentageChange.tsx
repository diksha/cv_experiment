import { IncidentTrend } from "features/dashboard";

import { Box, Typography, useTheme, TypographyProps, BoxProps } from "@mui/material";
import { ArrowDownward, ArrowUpward, TrendingUp, HorizontalRule } from "@mui/icons-material";
import { TimeBucketWidth } from "__generated__/globalTypes";
import { isToday } from "shared/utilities/dateutil";
import { DateTime } from "luxon";
import { StyledTooltip } from "ui";

interface PercentageChangeProps {
  trend: IncidentTrend;
  groupBy?: TimeBucketWidth;
  boxProps?: BoxProps;
  textProps?: TypographyProps;
}

export function PercentageChange({ trend, groupBy = TimeBucketWidth.DAY, boxProps, textProps }: PercentageChangeProps) {
  const theme = useTheme();
  const noPercent = trend.percentage === null || groupBy === TimeBucketWidth.HOUR;
  if (noPercent) {
    return (
      <StyledTooltip
        title={`Trends will appear when Voxel has detected at least 50 incidents and observed at least 1 month.`}
        placement="top"
      >
        <Box display="flex" {...boxProps}>
          <Box /> {/* text+icon combo in all cases for easier styling */}
          <TrendingUp
            sx={{
              color: theme.palette.text.dark,
              backgroundColor: theme.palette.grey[200],
              borderRadius: "50%",
              fontSize: 26,
              padding: "6px",
            }}
          />
        </Box>
      </StyledTooltip>
    );
  }
  const percent = trend.percentage as number;
  let groupsLength = trend.mergedOneDayGroups.length;
  let thePast = "";
  const lastGroupIsToday = isToday(
    DateTime.fromISO(trend.mergedOneDayGroups[groupsLength - 1].dimensions.datetime, { setZone: true })
  );
  if (lastGroupIsToday) {
    groupsLength--;
    thePast = "the past ";
  }

  const tooltipTitle =
    percent === 0
      ? `No change in daily incidents over ${thePast}${groupsLength} days`
      : `The trend of daily incidents over ${thePast}${groupsLength} days`;
  return (
    <StyledTooltip title={tooltipTitle} placement="top">
      <Box display="flex" {...boxProps}>
        {percent === 0 ? (
          <Box />
        ) : (
          <Typography paddingRight={0.5} variant="h4" {...textProps}>
            {`${Math.abs(percent)}%`}
          </Typography>
        )}
        {percent === 0 && (
          <HorizontalRule
            sx={{
              color: theme.palette.text.dark,
              backgroundColor: theme.palette.grey[200],
              borderRadius: "50%",
              fontSize: 26,
              padding: "6px",
            }}
          />
        )}
        {percent < 0 && (
          <ArrowDownward
            sx={{
              width: "20px",
              height: "20px",
              color: theme.palette.success[600],
            }}
          />
        )}
        {percent > 0 && (
          <ArrowUpward
            sx={{
              width: "20px",
              height: "20px",
              color: theme.palette.error[600],
            }}
          />
        )}
      </Box>
    </StyledTooltip>
  );
}
