import { INCIDENTS_HIGHLIGHT, INCIDENTS_PUBLIC_LINK_CREATE, INCIDENTS_RESOLVE, useCurrentUser } from "features/auth";
import { Transition } from "@headlessui/react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Player } from "features/video";
import classNames from "classnames";
import { ArrowSquareOut } from "phosphor-react";
import { RotateRight } from "@mui/icons-material";
import { DateTimeString, Spinner, getTimeRangeString } from "ui";
import { toHumanRead } from "features/analytics/helpers";
import { filterNullValues } from "shared/utilities/types";
import { Button } from "@mui/material";
import { useQuery } from "@apollo/client";
import { ActionPill, BrightYellow, PillActions, ResolveButton } from "features/incidents";
import {
  AssignButton,
  FeedbackButton,
  EventShareLinkButton,
  Highlight,
  Bookmark,
  GET_INCIDENT_DETAILS,
  SeenStateManager,
} from "features/incidents";
import { GetIncidentDetails, GetIncidentDetailsVariables } from "__generated__/GetIncidentDetails";
import { GetIncidentFeed_incidentFeed_edges_node } from "__generated__/GetIncidentFeed";
import { useRouting } from "shared/hooks";

interface IncidentRowProps {
  incident: GetIncidentFeed_incidentFeed_edges_node;
  active: boolean;
  uiKey?: string;
  showDuration?: boolean;
  mobileOnly?: boolean;
  onClick: () => void;
}

export function IncidentRow({ incident, active, uiKey, showDuration, mobileOnly, onClick }: IncidentRowProps) {
  const seen = useMemo(() => {
    return active || SeenStateManager.isSeen(incident.uuid);
  }, [incident, active]);

  const [actions, setActions] = useState<Array<PillActions>>([]);

  useEffect(() => {
    const pillActions: PillActions[] = [];

    if (incident.status === "resolved") {
      pillActions.push(PillActions.RESOLVED);
    } else {
      if (incident.priority?.toLowerCase() === "high") {
        pillActions.push(PillActions.HIGH_PRIORITY);
      }
      if (incident.assignees && incident.assignees?.length > 0) {
        pillActions.push(PillActions.ASSIGNED);
      }
    }

    if (incident.alerted) {
      pillActions.push(PillActions.ALERTED);
    }

    if (incident.highlighted) {
      pillActions.push(PillActions.HIGHLIGHTED);
    }

    if (seen) {
      pillActions.push(PillActions.SEEN);
    }

    setActions(pillActions);
  }, [incident, seen]);

  return (
    <div
      className="flex border-t first:border-t-0 hover:bg-gray-50 transition-colors duration-75"
      data-ui-key={uiKey ? `${uiKey}-incident-row` : "incident-row"}
    >
      <div className="flex-grow">
        {mobileOnly ? (
          <MobileIncidentRow
            onClick={onClick}
            incident={incident}
            actions={actions}
            showDuration={showDuration}
            showOnDesktop
          />
        ) : (
          <>
            <DesktopIncidentRow onClick={onClick} incident={incident} actions={actions} showDuration={showDuration} />
            <MobileIncidentRow onClick={onClick} incident={incident} actions={actions} showDuration={showDuration} />
          </>
        )}
        <Transition
          show={active}
          appear={true}
          enter="transition ease-in-out duration-200 delay-75"
          enterFrom="opacity-0"
          enterTo="opacity-100"
        >
          <IncidentDetails incidentUuid={incident.uuid!} />
        </Transition>
      </div>
    </div>
  );
}

interface DesktopIncidentRowProps {
  onClick: () => void;
  incident: GetIncidentFeed_incidentFeed_edges_node;
  actions: PillActions[];
  showDuration?: boolean;
}

function DesktopIncidentRow({ onClick, incident, actions, showDuration }: DesktopIncidentRowProps) {
  const { currentUser } = useCurrentUser();

  const useDuration =
    showDuration && incident.incidentType?.key === "PRODUCTION_LINE_DOWN" && incident.duration && incident.endTimestamp;

  const duration = useDuration && incident.duration ? toHumanRead(incident.duration) : "";
  const timeRange = useDuration
    ? getTimeRangeString(incident.timestamp, incident.endTimestamp, currentUser?.site?.timezone)
    : "";

  return (
    <div className="hidden md:flex items-stretch gap-4 p-4 cursor-pointer z-10" onClick={onClick}>
      <div
        className="w-24 h-16 rounded-md bg-cover bg-brand-gray-800 bg-center overflow-hidden"
        style={{ backgroundImage: `url(${incident?.camera?.thumbnailUrl})` }}
      ></div>
      <div className="flex flex-col flex-1">
        <div className="px-0 md:static flex justify-between flex-grow">
          <div className="flex flex-col justify-between">
            <div className="flex items-center">
              <div className="font-bold text-sm mx-2">{incident.incidentType?.name}</div>
            </div>
            {incident.timestamp && !useDuration ? (
              <div className="text-sm mx-2 text-brand-gray-300">
                <DateTimeString dateTime={incident.timestamp} format={"ccc, LLL d, h:mm"} includeTimezone={true} />
              </div>
            ) : null}
            {useDuration ? (
              <>
                <div className="mx-2 text-brand-gray-300">{timeRange}</div>
                <div className="mx-2">
                  <div className="flex items-center">
                    <RotateRight
                      sx={{
                        width: "20px",
                        height: "20px",
                      }}
                    />
                    <div className="text-sm text-brand-gray-300">{duration}</div>
                  </div>
                </div>
              </>
            ) : (
              <div className="text-sm mx-2 text-brand-gray-300">{incident.camera?.name}</div>
            )}
          </div>
          <div className="flex items-center">
            <div className="flex">
              {actions.map((action) => {
                return <ActionPill key={action + incident.id!} type={action} collapsable={true} />;
              })}
            </div>
            {!currentUser?.isDemoEnvironment && currentUser?.hasGlobalPermission(INCIDENTS_HIGHLIGHT) ? (
              <Highlight incidentId={incident.id} highlighted={incident.highlighted} />
            ) : null}
            <Bookmark incidentId={incident.id} bookmarked={incident.bookmarked} />
          </div>
        </div>
      </div>
    </div>
  );
}

interface MobileIncidentRowProps {
  onClick: () => void;
  incident: GetIncidentFeed_incidentFeed_edges_node;
  actions: PillActions[];
  showDuration?: boolean;
  showOnDesktop?: boolean;
}

function MobileIncidentRow({ onClick, incident, actions, showDuration, showOnDesktop }: MobileIncidentRowProps) {
  const { currentUser } = useCurrentUser();

  const useDuration =
    showDuration && incident.incidentType?.key === "PRODUCTION_LINE_DOWN" && incident.duration && incident.endTimestamp;

  const duration = useDuration && incident.duration ? toHumanRead(incident.duration) : "";
  const timeRange = useDuration
    ? getTimeRangeString(incident.timestamp, incident.endTimestamp, currentUser?.site?.timezone)
    : "";

  return (
    <div
      className={classNames("flex items-stretch gap-4 p-4 cursor-pointer", !showOnDesktop && "md:hidden")}
      onClick={onClick}
    >
      <div
        className="w-24 h-16 rounded-md bg-cover bg-brand-gray-800 bg-center overflow-hidden"
        style={{ backgroundImage: `url(${incident?.camera?.thumbnailUrl})` }}
      ></div>
      <div className="flex flex-col flex-1">
        <div className="px-0 md:static flex justify-between flex-grow">
          <div className={classNames("flex flex-col", useDuration ? "justify-around" : "justify-between")}>
            <div className="font-bold text-md mx-2">{incident.incidentType?.name}</div>
            {incident.timestamp && !useDuration ? (
              <div className="text-sm mx-2 text-brand-gray-300">
                <DateTimeString dateTime={incident.timestamp} format={"ccc, LLL d, h:mm"} />
              </div>
            ) : incident.duration ? (
              <>
                <div className="mx-2 text-brand-gray-300">{timeRange}</div>
                <div className="mx-2 flex items-center">
                  <RotateRight
                    sx={{
                      width: "20px",
                      height: "20px",
                    }}
                  />
                  <div className="text-sm text-brand-gray-300">{duration}</div>
                </div>
              </>
            ) : null}
            {!useDuration ? <div className="text-sm mx-2 text-brand-gray-300">{incident.camera?.name}</div> : null}
          </div>
          <div className="flex items-center">
            <div className="flex gap-2">
              {actions.map((action) => {
                return <ActionPill key={action + incident.id!} type={action} showText={false} />;
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// TODO: The IncidentDetails is the same as the IncidentDetails component under
// services/portal/web/src/features/incidents/components/table/TableRow.tsx
// need to put them into a shared component
function IncidentDetails(props: { incidentUuid: string }) {
  const { currentUser } = useCurrentUser();
  const { newLocationState } = useRouting();

  const { loading, data, refetch } = useQuery<GetIncidentDetails, GetIncidentDetailsVariables>(GET_INCIDENT_DETAILS, {
    variables: {
      incidentUuid: props.incidentUuid,
    },
  });

  const assignees = data?.incidentDetails?.assignees || [];
  const assignedUserIds = filterNullValues<string>(assignees.map((assignee) => assignee?.id));

  useEffect(() => {
    SeenStateManager.markAsSeen(props.incidentUuid);
  }, [props.incidentUuid]);

  return (
    <>
      {loading ? (
        <div className="grid p-36 justify-center opacity-40">
          <div>
            <Spinner />
          </div>
        </div>
      ) : null}
      {data?.incidentDetails ? (
        <div className="grid grid-cols-1 gap-4 p-4 pt-0 overflow-hidden lg:grid-cols-4">
          <div className="col-span-1 lg:col-span-3">
            <Player
              videoUrl={data.incidentDetails.videoUrl!}
              annotationsUrl={data.incidentDetails.annotationsUrl}
              actorIds={data.incidentDetails.actorIds}
              annotationColorHex={BrightYellow}
              controls
            />
          </div>
          <div className="col-span-1">
            <div className="flex flex-col gap-y-2 text-sm text-center">
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
                  incidentTitle={data.incidentDetails.incidentType?.name!}
                />
              )}
              <FeedbackButton
                data-ui-key="button-provide-feedback"
                incidentId={data.incidentDetails.id!}
                incidentTitle={data.incidentDetails.incidentType?.name!}
              />
              <Button
                id={data.incidentDetails.id}
                data-ui-key="button-view-incident-details"
                variant="outlined"
                component={Link}
                to={`/incidents/${data.incidentDetails.uuid}`}
                state={newLocationState}
                startIcon={<ArrowSquareOut />}
              >
                View Details
              </Button>
            </div>
            <div className="pt-4">
              <AssignButton
                incidentId={data.incidentDetails.id}
                incidentUuid={data.incidentDetails.uuid!}
                incidentTitle={data.incidentDetails.incidentType?.name!}
                assignedUserIds={assignedUserIds}
              />
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
