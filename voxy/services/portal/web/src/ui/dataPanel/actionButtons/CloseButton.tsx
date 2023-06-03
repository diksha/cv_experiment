import { Close, ArrowBack, ArrowForward, ArrowDownward, SvgIconComponent } from "@mui/icons-material";
import { IconButton, useTheme } from "@mui/material";
import { ACTION_BUTTON_WIDTH, ACTION_BUTTON_PADDING, ACTION_BUTTON_ICON_WIDTH } from "./constants";

export type CloseIconVariant = "x" | "arrowLeft" | "arrowDown" | "arrowRight";

const CloseIcons: { [key in CloseIconVariant]: SvgIconComponent } = {
  x: Close,
  arrowLeft: ArrowBack,
  arrowRight: ArrowForward,
  arrowDown: ArrowDownward,
};

interface CloseButtonProps {
  onClick: () => void;
  variant?: CloseIconVariant;
}
export function CloseButton({ onClick, variant }: CloseButtonProps) {
  const theme = useTheme();
  const Icon = CloseIcons[variant || "x"];

  return (
    <IconButton
      onClick={onClick}
      sx={{
        display: "block",
        backgroundColor: theme.palette.primary[700],
        color: theme.palette.grey[100],
        width: ACTION_BUTTON_WIDTH,
        height: ACTION_BUTTON_WIDTH,
        padding: ACTION_BUTTON_PADDING,
        "&:hover": {
          backgroundColor: theme.palette.primary[900],
        },
      }}
    >
      <Icon
        sx={{
          height: ACTION_BUTTON_ICON_WIDTH,
          width: ACTION_BUTTON_ICON_WIDTH,
        }}
      />
    </IconButton>
  );
}
