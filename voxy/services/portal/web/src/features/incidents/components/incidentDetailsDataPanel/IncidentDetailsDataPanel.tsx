import {
  INCIDENTS_HIGHLIGHT,
  useCurrentUser,
  INCIDENTS_PUBLIC_LINK_CREATE,
  CurrentUser,
  INCIDENTS_RESOLVE,
} from "features/auth";
import { filterNullValues } from "shared/utilities/types";
import { useMediaQuery, Stack, Typography, Box, useTheme } from "@mui/material";
import {
  AssignButton,
  Bookmark,
  Highlight,
  ExportVideo,
  EventShareLinkButton,
  GET_INCIDENT_DETAILS,
  IncidentActivity,
  ResolveButton,
  FeedbackButton,
  IncidentTag,
  StatusActionPill,
  ActionPill,
  PillActions,
  BrightYellow,
} from "features/incidents";
import React, { useMemo } from "react";
import { useQuery } from "@apollo/client";
import { IncidentDetailsDataPanelError } from "./IncidentDetailsDataPanelError";
import { IncidentDetailsDataPanelSkeleton } from "./IncidentDetailsDataPanelSkeleton";

import { DateTimeString, getDateTimeRangeString, DataPanel } from "ui";
import { toHumanRead } from "features/analytics/helpers";
import { Player } from "features/video";
import {
  GetIncidentDetails,
  GetIncidentDetailsVariables,
  GetIncidentDetails_incidentDetails,
} from "__generated__/GetIncidentDetails";

interface IncidentDetails extends GetIncidentDetails_incidentDetails {}

interface IncidentAttributeListItem {
  label: string;
  value: string | React.ReactNode;
}

interface IncidentDetailsDataPanelProps {
  incidentUuid: string;
  open: boolean;
  onClose: () => void;
}
export function IncidentDetailsDataPanel({ open, onClose, ...props }: IncidentDetailsDataPanelProps) {
  const theme = useTheme();
  const xsBreakpoint = useMediaQuery(theme.breakpoints.down("sm"));
  return (
    <DataPanel open={open} onClose={onClose} closeIconVariant={xsBreakpoint ? "arrowDown" : "arrowRight"}>
      {open ? <IncidentDetailsDataPanelContent {...props} /> : null}
    </DataPanel>
  );
}

type IncidentDetailsDataPanelContentProps = Omit<IncidentDetailsDataPanelProps, "open" | "onClose">;

function IncidentDetailsDataPanelContent({ incidentUuid }: IncidentDetailsDataPanelContentProps) {
  const { currentUser } = useCurrentUser();
  const theme = useTheme();
  const { loading, error, data, refetch } = useQuery<GetIncidentDetails, GetIncidentDetailsVariables>(
    GET_INCIDENT_DETAILS,
    {
      variables: {
        incidentUuid,
      },
    }
  );

  const attributeList: IncidentAttributeListItem[] = useMemo(() => {
    if (data?.incidentDetails) {
      return buildAttributeList(data.incidentDetails, currentUser);
    }
    return [];
  }, [data, currentUser]);

  if (loading) {
    return <IncidentDetailsDataPanelSkeleton />;
  }

  if (error || !data?.incidentDetails) {
    return <IncidentDetailsDataPanelError />;
  }

  const incidentTypeName = data.incidentDetails.incidentType?.name || "Unknown";
  const timestampString =
    data.incidentDetails.incidentType?.key === "PRODUCTION_LINE_DOWN" && data.incidentDetails.duration ? (
      <span>
        {getDateTimeRangeString(
          data.incidentDetails.timestamp,
          data.incidentDetails.endTimestamp,
          currentUser?.site?.timezone
        )}
        <span>&nbsp;({toHumanRead(data.incidentDetails.duration)})</span>
      </span>
    ) : (
      <DateTimeString dateTime={data.incidentDetails.timestamp} includeTimezone />
    );
  const actorIds = filterNullValues<string>(data.incidentDetails?.actorIds);

  const assignees = data?.incidentDetails?.assignees || [];
  const assignedUserIds = filterNullValues<string>(assignees.map((assignee) => assignee?.id));

  return (
    <Stack spacing={2} data-ui-key="incident-details-data-panel">
      <Box display="flex" gap="1rem" alignItems="center">
        <Box flex="1">
          <Typography variant="h3" fontWeight={800} paddingBottom={1}>
            {incidentTypeName}
          </Typography>
          <Typography>{timestampString}</Typography>
        </Box>
        <Box display="flex" gap=".5rem">
          {!currentUser?.isDemoEnvironment && currentUser?.hasGlobalPermission(INCIDENTS_HIGHLIGHT) ? (
            <Highlight incidentId={data.incidentDetails.id} highlighted={data.incidentDetails.highlighted} />
          ) : null}
          <Bookmark incidentId={data.incidentDetails.id} bookmarked={!!data.incidentDetails.bookmarked} />
          <ExportVideo incidentId={data.incidentDetails.id} />
        </Box>
      </Box>
      <Box>
        <Player
          videoUrl={data.incidentDetails.videoUrl!}
          annotationsUrl={data.incidentDetails.annotationsUrl!}
          actorIds={actorIds}
          annotationColorHex={BrightYellow}
          controls
        />
      </Box>
      <Box display="flex" flexDirection={{ xs: "column", sm: "row" }} gap="1rem" justifyContent="space-between">
        <Box flex="1" border="1px solid" borderColor={theme.palette.grey[300]} borderRadius="8px">
          {attributeList.map((item) => (
            <div className="flex p-4 gap-2 border-b border-b-brand-gray-050 last:border-b-0" key={item.label}>
              <div className="font-bold">{item.label}:</div>
              <div className="flex-grow">{item.value}</div>
            </div>
          ))}
        </Box>
        <Box flex="1" display="flex" flexDirection="column" gap=".5rem">
          {currentUser?.hasZonePermission(INCIDENTS_RESOLVE) && (
            <ResolveButton
              incidentId={data.incidentDetails.id}
              resolved={data.incidentDetails.status?.toLowerCase() === "resolved"}
              fullWidth
              onReopen={() => refetch()}
              onResolve={() => refetch()}
            />
          )}
          {currentUser?.hasZonePermission(INCIDENTS_PUBLIC_LINK_CREATE) && (
            <EventShareLinkButton
              incidentId={data.incidentDetails.id}
              incidentTitle={data.incidentDetails.incidentType?.name || ""}
              fullWidth
            />
          )}
          <FeedbackButton
            data-ui-key="button-provide-feedback"
            incidentId={data.incidentDetails.id}
            incidentTitle={data.incidentDetails.incidentType?.name}
            fullWidth
          />
          <AssignButton
            incidentId={data.incidentDetails.id}
            incidentUuid={data.incidentDetails.uuid}
            incidentTitle={data.incidentDetails.incidentType?.name || ""}
            assignedUserIds={assignedUserIds}
          />
        </Box>
      </Box>
      <Box>
        <IncidentActivity
          incidentId={data.incidentDetails.id}
          incidentUuid={data.incidentDetails.uuid}
          incidentTimestamp={data.incidentDetails.timestamp}
        />
      </Box>
    </Stack>
  );
}

function buildAttributeList(
  incidentDetails: IncidentDetails,
  currentUser: CurrentUser | undefined
): IncidentAttributeListItem[] {
  const list: IncidentAttributeListItem[] = [];
  const siteName = incidentDetails.zone?.name;
  if (siteName) {
    list.push({ label: "Zone", value: siteName });
  }

  const cameraName = incidentDetails.camera?.name;
  if (cameraName) {
    list.push({ label: "Camera", value: cameraName });
  }

  const incidentTypeName = incidentDetails.incidentType?.name;
  if (incidentTypeName) {
    list.push({ label: "Type", value: incidentTypeName });
  }

  const showDuration = incidentDetails.duration && incidentDetails?.incidentType?.key === "PRODUCTION_LINE_DOWN";

  if (showDuration) {
    list.push({ label: "Duration", value: toHumanRead(incidentDetails.duration!) });
  }

  if (incidentDetails.tags && incidentDetails.tags.length > 0) {
    const tags: React.ReactElement[] = incidentDetails.tags
      ?.filter((tag) => !!tag)
      .map((tag) => (
        <IncidentTag
          key={`${tag?.label}-${tag?.value}`}
          label={tag?.label || "Unknown"}
          value={tag?.value || "Unknown"}
        />
      ));

    tags.push(
      <StatusActionPill key="priority-status" priority={incidentDetails.priority} status={incidentDetails.status} />
    );

    if (incidentDetails.alerted) {
      tags.push(<ActionPill type={PillActions.ALERTED} />);
    }

    if (incidentDetails.highlighted) {
      tags.push(<ActionPill type={PillActions.HIGHLIGHTED} />);
    }

    list.push({
      label: "Tags",
      value: <div className="flex flex-wrap gap-2">{tags.map((tag) => tag)}</div>,
    });
  }

  return list;
}
