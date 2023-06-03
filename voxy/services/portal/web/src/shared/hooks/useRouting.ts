import { PAGE_EXECUTIVE_DASHBOARD, useCurrentUser } from "features/auth";
import { useLocation } from "react-router-dom";

interface LocationState {
  visitedDash?: string;
}
type RouterLocationState = LocationState | undefined; // state on <Link> is of type unknown

export function useRouting() {
  const { currentUser } = useCurrentUser();
  const location = useLocation();
  const locationState = location.state as RouterLocationState;
  const fromExec =
    locationState?.visitedDash === "/executive-dashboard" ||
    (locationState?.visitedDash === undefined && currentUser?.hasGlobalPermission(PAGE_EXECUTIVE_DASHBOARD));

  return {
    location,
    locationState,
    // TODO(hq): find cleaner alternative for tracking last visited dash than repeatedly needing to update location state
    newLocationState: {
      visitedDash: ["/dashboard", "/executive-dashboard"].includes(location.pathname)
        ? location.pathname
        : locationState?.visitedDash,
    },
    fromExec,
  };
}
