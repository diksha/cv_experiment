import { useState, useMemo, useCallback } from "react";
import { useQuery } from "@apollo/client";
import { DateTime } from "luxon";
import { Spinner } from "ui";
import { IncidentDetailsDataPanel } from "features/incidents";
import { GET_CURRENT_ZONE_ACTIVITY } from "features/activity";
import { ChatText, Checks, Users } from "phosphor-react";
import {
  GetCurrentZoneActivity,
  GetCurrentZoneActivity_currentUser_zone_recentComments_edges_node,
} from "__generated__/GetCurrentZoneActivity";
import { getNodes } from "graphql/utils";
import { Avatar, Box, Button, Grid, Typography, capitalize, useTheme } from "@mui/material";
import { useDashboardContext } from "features/dashboard/hooks/dashboard";
interface Comment extends GetCurrentZoneActivity_currentUser_zone_recentComments_edges_node {}

const ASSIGNED_TO_PATTERN = "assigned to";
const UNASSIGNED_BY = "unassigned by";
const RESOLVED_BY = "resolved by";
const ASSIGN = "assign";
const UNASSIGN = "unassigned";
const RESOLVE = "resolve";
const COMMENT = "comment";

enum ActivityStyle {
  DEFAULT = "DEFAULT",
  INCIDENT_RESOLVED = "INCIDENT_RESOLVED",
}

interface IncidentActivity {
  id: string;
  linkTo: string;
  style: ActivityStyle;
  incidentUuid: string;
  incidentTitle: string;
  incidentPriority: string;
  incidentStatus: string | null;
  thumbnailUrl: string;
  ownerFullName: string;
  ownerInitials: string;
  ownerPicture?: string | null;
  message: string;
  messageType: string;
  timestamp: DateTime;
}

function extractMessage(comment: Comment): { type: string; text: string } {
  // This is a hacky temp solution.
  // Introducing a proper activity type instead of using comments.
  const { incident, text, owner } = comment;
  const eventName = incident?.incidentType?.name || "";
  const ownerFullName = owner?.fullName || "Someone";
  if (text.includes(ASSIGNED_TO_PATTERN)) {
    const newText = text.replace("to ", "");
    return { type: ASSIGN, text: `${newText} a ${eventName} incident` };
  }
  if (text.includes(UNASSIGNED_BY)) {
    const newText = text.replace("by ", "");
    return { type: UNASSIGN, text: `${newText} a ${eventName} incident` };
  }
  if (text.includes(RESOLVED_BY)) {
    if (eventName === "") {
      return { type: RESOLVE, text: `Incident ${text}` };
    }
    return { type: RESOLVE, text: `${capitalize(eventName)} incident ${text}` };
  }
  return { type: COMMENT, text: `${ownerFullName} commented, "${text}"` };
}

function getMessageIconComponent(type: string): JSX.Element {
  if (type === RESOLVE) {
    return <Checks size={28} weight="light" />;
  }
  if (type === ASSIGN) {
    return <Users size={28} weight="light" />;
  }
  if (type === UNASSIGN) {
    return <Users size={28} weight="light" />;
  }
  return <ChatText size={28} weight="light" />;
}

export function CurrentZoneActivityFeed(): JSX.Element {
  const theme = useTheme();
  const { feedSlideOpen, setFeedSlideOpen } = useDashboardContext();
  const { data, loading, fetchMore } = useQuery<GetCurrentZoneActivity>(GET_CURRENT_ZONE_ACTIVITY, {
    notifyOnNetworkStatusChange: true,
    variables: {
      activityItemsPerFetch: 10,
    },
  });
  const activities: IncidentActivity[] = useMemo(() => {
    const results: IncidentActivity[] = [];
    getNodes<Comment>(data?.currentUser?.zone?.recentComments)
      .sort((a: Comment, b: Comment) => -1 * a.createdAt.localeCompare(b.createdAt)) // DESC
      .forEach((comment) => {
        const ownerFullName = comment.owner?.fullName || "Someone";
        const { text: message, type: messageType } = extractMessage(comment);
        if (comment.incident) {
          results.push({
            id: comment.id,
            linkTo: `/incidents/${comment.incident?.uuid || "unknown-incident"}`,
            style: ActivityStyle.DEFAULT,
            incidentUuid: comment.incident.uuid,
            incidentTitle: comment.incident.title,
            incidentPriority: comment.incident.priority,
            incidentStatus: comment.incident.status,
            thumbnailUrl: comment.incident.thumbnailUrl,
            ownerFullName,
            ownerInitials: comment.owner?.initials || "?",
            ownerPicture: comment.owner?.picture,
            message,
            messageType,
            timestamp: DateTime.fromISO(comment.createdAt),
          });
        }
      });
    return results;
  }, [data]);

  const handleLoadMore = () => {
    fetchMore({
      variables: {
        activityAfter: data?.currentUser?.zone?.recentComments?.pageInfo.endCursor,
      },
    });
  };

  const handleUseDrawer = useCallback(() => {
    setFeedSlideOpen(true);
  }, [setFeedSlideOpen]);

  const hasMore = !!data && data?.currentUser?.zone?.recentComments?.pageInfo?.hasNextPage;
  const useDrawer = !feedSlideOpen;
  const size = !feedSlideOpen ? 10 : activities.length;

  return (
    <div data-ui-key="activity-feed">
      {activities.slice(0, size).map((activity) => (
        <ActivityItem key={activity.id} activity={activity} />
      ))}
      {loading ? (
        <Grid container justifyContent="center" alignItems="center">
          <Box px={4} py={8}>
            <Spinner className="mx-auto opacity-40" />
          </Box>
        </Grid>
      ) : null}
      {!loading && hasMore ? (
        <Box p={2}>
          <Button
            variant="outlined"
            onClick={useDrawer ? handleUseDrawer : handleLoadMore}
            sx={{ width: "100%" }}
            data-ui-key="activity-feed-see-more"
          >
            See more
          </Button>
        </Box>
      ) : null}
      {!loading && activities.length === 0 ? (
        <Box p={2}>
          <Typography align="center" color={theme.palette.grey[500]}>
            No recent activity
          </Typography>
        </Box>
      ) : null}
    </div>
  );
}

interface ActivityItemProps {
  activity: IncidentActivity;
}
function ActivityItem({ activity }: ActivityItemProps) {
  const theme = useTheme();
  const [dataPanelOpen, setDataPanelOpen] = useState(false);

  const handleClick = () => {
    setDataPanelOpen(true);
  };

  const handleClose = () => {
    setDataPanelOpen(false);
  };
  return (
    <>
      <Button
        onClick={handleClick}
        key={activity.id}
        sx={{
          display: "flex",
          width: "100%",
          textAlign: "left",
          textTransform: "none",
          alignItems: "start",
          gap: 2,
          paddingY: 2,
          paddingX: 1,
          borderRadius: "8px",
          ":hover": {
            backgroundColor: theme.palette.grey[200],
          },
        }}
        data-ui-key="datapanel-activity-incident"
      >
        <Box sx={{ alignItems: "center" }}>
          <Avatar sx={{ bgcolor: theme.palette.grey[200], position: "static" }}>
            {getMessageIconComponent(activity.messageType)}
          </Avatar>
        </Box>
        <Box width={"100%"}>
          <Typography
            sx={{
              textOverflow: "ellipsis",
              display: "-webkit-box",
              WebkitLineClamp: "3",
              WebkitBoxOrient: "vertical",
              overflow: "hidden",
            }}
          >
            {activity.message}
          </Typography>
          <Typography color={theme.palette.grey[500]}>{activity.timestamp.toRelative()}</Typography>
        </Box>
      </Button>
      <IncidentDetailsDataPanel incidentUuid={activity.incidentUuid} open={dataPanelOpen} onClose={handleClose} />
    </>
  );
}
