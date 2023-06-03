import { Tooltip, TooltipProps, styled, tooltipClasses } from "@mui/material";

const StyledTooltip = styled(({ className, ...props }: TooltipProps) => (
  <Tooltip {...props} classes={{ popper: className }} />
))({
  [`& .${tooltipClasses.tooltip}`]: {
    maxWidth: 200,
    fontSize: 14,
  },
});

export { StyledTooltip };
