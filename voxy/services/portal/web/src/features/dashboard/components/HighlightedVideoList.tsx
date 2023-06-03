import { GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node } from "__generated__/GetCurrentZoneIncidentFeed";
import { useMediaQuery, useTheme } from "@mui/material";
import { VideoCard } from "./VideoCard";

interface HighlightedVideoListProps {
  items: GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node[];
}
export function HighlightedVideoList({ items }: HighlightedVideoListProps) {
  // Taking a shortcut since the query is a temp solution. Refactor it if possible
  const theme = useTheme();
  const mdBreakpoint = useMediaQuery(theme.breakpoints.up("md"));

  const size = mdBreakpoint ? 4 : 1;
  const array: JSX.Element[] = [];
  items.forEach((item) => {
    // Only showing the "DailyIncidentsFeedItem" case
    if (item.__typename === "DailyIncidentsFeedItem") {
      if (item.timeBuckets) {
        item.timeBuckets
          .filter((timeBucket) => timeBucket && timeBucket?.incidentCount > 0)
          .reverse()
          .forEach((timeBucket) => {
            timeBucket?.latestIncidents?.forEach((incident) => {
              if (array.length < size && !!incident) {
                array.push(<VideoCard key={incident.id} incident={incident} />);
              }
            });
          });
      }
    }
  });
  return <>{array}</>;
}
