import { ElementType } from "react";
import Popup from "reactjs-popup";
import { Link } from "react-router-dom";
import { useTheme, Box, useMediaQuery } from "@mui/material";
import { NAV_ITEM_WIDTH, NAV_ITEM_ICON_WIDTH } from "./constants";
import { isCurrentPath } from "./helpers";
import { useRouting } from "shared/hooks";

export function NavItem(props: {
  to: string;
  text?: string;
  extraMatches?: Array<string>;
  icon: ElementType;
  uiKey?: string;
}) {
  const theme = useTheme();
  const { location, newLocationState } = useRouting();
  const active = isCurrentPath(location.pathname, props.to, props.extraMatches);
  const mdBreakpoint = useMediaQuery(theme.breakpoints.up("md"));

  return (
    <Popup
      trigger={
        <Link to={props.to} data-ui-key={props.uiKey} state={newLocationState}>
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: NAV_ITEM_WIDTH,
              height: NAV_ITEM_WIDTH,
              borderRadius: 2,
              backgroundColor: active ? theme.palette.grey[300] : "transparent",
              "&:hover": { backgroundColor: theme.palette.grey[300] },
            }}
          >
            <props.icon sx={{ height: NAV_ITEM_ICON_WIDTH, width: NAV_ITEM_ICON_WIDTH }} />
          </Box>
        </Link>
      }
      contentStyle={{ background: "#262626", color: "white", border: "0", zIndex: theme.zIndex.drawer + 1 }}
      arrowStyle={{ color: "#262626" }}
      on={["hover", "focus"]}
      arrow={true}
      mouseEnterDelay={1000}
      position={["right center"]}
      closeOnDocumentClick
      disabled={!mdBreakpoint}
    >
      <Box component="span" sx={{ fontWeight: "bold", paddingX: 2 }}>
        {props.text}
      </Box>
    </Popup>
  );
}
