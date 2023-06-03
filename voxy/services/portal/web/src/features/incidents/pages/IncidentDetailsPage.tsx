import { INCIDENTS_HIGHLIGHT, useCurrentUser, INCIDENTS_PUBLIC_LINK_CREATE, INCIDENTS_RESOLVE } from "features/auth";
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
  SeenStateManager,
  StatusActionPill,
  ActionPill,
  PillActions,
  BrightYellow,
} from "features/incidents";
import React, { useEffect, useMemo } from "react";
import { filterNullValues } from "shared/utilities/types";
import classNames from "classnames";
import { useQuery } from "@apollo/client";
import { useNavigate, useParams } from "react-router-dom";

import { PageWrapper, Card, DateTimeString } from "ui";
import { Player } from "features/video";
import { CaretLeft } from "phosphor-react";
import { GetIncidentDetails, GetIncidentDetailsVariables } from "__generated__/GetIncidentDetails";
import { toHumanRead } from "features/analytics/helpers";

export const INCIDENT_DETAILS_PATH = "/incidents/:incidentUuid";

interface IncidentAttributeListItem {
  label: string;
  value: string | React.ReactNode;
}

export function IncidentDetailsPage() {
  const { currentUser } = useCurrentUser();
  const { incidentUuid }: any = useParams();
  const navigate = useNavigate();
  const { loading, error, data, refetch } = useQuery<GetIncidentDetails, GetIncidentDetailsVariables>(
    GET_INCIDENT_DETAILS,
    {
      variables: {
        incidentUuid,
      },
    }
  );

  useEffect(() => {
    SeenStateManager.markAsSeen(incidentUuid);
  }, [incidentUuid]);

  const handleBack = () => {
    if (window.history.length > 1) {
      navigate(-1);
    } else {
      // If this page was cold-loaded, redirect to user-specific home page
      navigate("/");
    }
  };

  const attributeList: IncidentAttributeListItem[] = useMemo(() => {
    const list: IncidentAttributeListItem[] = [];
    if (data?.incidentDetails?.organization?.name) {
      list.push({
        label: "Organization",
        value: data.incidentDetails.organization.name,
      });
    }

    if (data?.incidentDetails?.zone?.name) {
      list.push({
        label: "Zone",
        value: data.incidentDetails.zone.name,
      });
    }

    if (data?.incidentDetails?.camera?.name) {
      list.push({
        label: "Camera",
        value: data.incidentDetails.camera.name,
      });
    }

    if (data?.incidentDetails?.incidentType?.name) {
      list.push({
        label: "Type",
        value: data.incidentDetails.incidentType.name,
      });
    }

    const showDuration =
      data?.incidentDetails?.duration && data?.incidentDetails?.incidentType?.key === "PRODUCTION_LINE_DOWN";

    if (showDuration) {
      list.push({ label: "Duration", value: toHumanRead(data?.incidentDetails?.duration!) });
    }

    if (data?.incidentDetails?.tags && data.incidentDetails.tags.length > 0) {
      const tags: React.ReactElement[] = data?.incidentDetails?.tags
        ?.filter((tag) => !!tag)
        .map((tag) => (
          <IncidentTag
            key={`${tag?.label}-${tag?.value}`}
            label={tag?.label || "Unknown"}
            value={tag?.value || "Unknown"}
          />
        ));

      tags.push(
        <StatusActionPill
          key="priority-status"
          priority={data.incidentDetails?.priority}
          status={data.incidentDetails?.status}
        />
      );

      if (data?.incidentDetails?.alerted) {
        tags.push(<ActionPill type={PillActions.ALERTED} />);
      }

      if (data?.incidentDetails?.highlighted) {
        tags.push(<ActionPill type={PillActions.HIGHLIGHTED} />);
      }

      list.push({
        label: "Tags",
        value: <div className="flex flex-wrap gap-2">{tags.map((tag) => tag)}</div>,
      });
    }

    return list;
  }, [data]);

  const actorIds: string[] = useMemo(() => {
    const ids = data?.incidentDetails?.actorIds || [];
    return ids.filter((id) => !!id) as string[];
  }, [data]);
  const assignees = data?.incidentDetails?.assignees || [];
  const assignedUserIds = filterNullValues<string>(assignees.map((assignee) => assignee?.id));

  return (
    <PageWrapper>
      <Card noPadding loading={loading} className={classNames({ "py-24": loading })}>
        {!loading && error ? (
          <div className="text-brand-gray-300 py-24 text-center">Failed to fetch incident details</div>
        ) : null}
        {!loading && data && data.incidentDetails ? (
          <div className="p-4 md:px-8">
            <div className="flex gap-4 items-center">
              <button className="p-2 -m-2 hover:bg-brand-gray-050 rounded-full" onClick={handleBack}>
                <CaretLeft className="h-6 w-6 text-brand-gray-300" />
              </button>
              <div className="flex-grow text-brand-gray-500">
                <div className="font-epilogue font-bold text-lg">
                  {data.incidentDetails?.incidentType?.name || "Unknown"}
                </div>
                <div className="text-md">
                  {data.incidentDetails?.timestamp ? (
                    <DateTimeString dateTime={data.incidentDetails.timestamp} includeTimezone />
                  ) : null}
                </div>
              </div>
              {data?.incidentDetails?.id ? (
                <>
                  {!currentUser?.isDemoEnvironment && currentUser?.hasGlobalPermission(INCIDENTS_HIGHLIGHT) ? (
                    <Highlight incidentId={data.incidentDetails.id} highlighted={data.incidentDetails.highlighted} />
                  ) : null}
                  <Bookmark incidentId={data.incidentDetails.id} bookmarked={!!data.incidentDetails.bookmarked} />
                  <ExportVideo incidentId={data.incidentDetails.id} />
                </>
              ) : null}
            </div>
            <div className="grid grid-cols-1 gap-4 pt-4 md:gap-8 lg:grid-cols-3">
              <div className="lg:col-span-2">
                <Player
                  videoUrl={data.incidentDetails.videoUrl!}
                  annotationsUrl={data.incidentDetails.annotationsUrl!}
                  actorIds={actorIds}
                  annotationColorHex={BrightYellow}
                  controls
                />
                <div className="pt-4"></div>
              </div>
              <div>
                <div className="flex flex-col gap-4">
                  <div className="flex flex-col border rounded-lg border-brand-gray-050">
                    {attributeList.map((item) => (
                      <div className="flex p-4 gap-2 border-b border-b-brand-gray-050 last:border-b-0" key={item.label}>
                        <div className="font-bold">{item.label}:</div>
                        <div className="flex-grow">{item.value}</div>
                      </div>
                    ))}
                  </div>
                  <div className="flex flex-col gap-2">
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
                        incidentId={data.incidentDetails.id!}
                        incidentTitle={data.incidentDetails?.incidentType?.name || ""}
                        fullWidth
                      />
                    )}
                    <FeedbackButton
                      data-ui-key="button-provide-feedback"
                      incidentId={data.incidentDetails.id!}
                      incidentTitle={data.incidentDetails.incidentType?.name!}
                      fullWidth
                    />
                  </div>
                  <div>
                    <AssignButton
                      incidentId={data.incidentDetails.id!}
                      incidentUuid={data.incidentDetails.uuid!}
                      incidentTitle={data.incidentDetails?.incidentType?.name || ""}
                      assignedUserIds={assignedUserIds}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : null}
      </Card>
      <div className="pt-8">
        <Card className={classNames("w-full", { "py-24": loading })} loading={loading}>
          {data?.incidentDetails?.uuid ? (
            <>
              <IncidentActivity
                incidentId={data.incidentDetails.id}
                incidentUuid={data.incidentDetails.uuid}
                incidentTimestamp={data.incidentDetails?.timestamp}
              />
            </>
          ) : null}
          {!loading && error ? (
            <div className="py-16 text-center text-brand-gray-300">Failed to load incident activity</div>
          ) : null}
        </Card>
      </div>
    </PageWrapper>
  );
}
