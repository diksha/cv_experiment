import { Transition } from "@headlessui/react";
import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Player } from "features/video";
import classNames from "classnames";
import { ArrowSquareOut } from "phosphor-react";
import { filterNullValues } from "shared/utilities/types";
import { Spinner, DateTimeString } from "ui";
import { Button } from "@mui/material";
import { useQuery } from "@apollo/client";
import {
  AssignButton,
  FeedbackButton,
  Bookmark,
  GET_INCIDENT_DETAILS,
  SeenStateManager,
  StatusActionPill,
  BrightYellow,
} from "features/incidents";

import { GetIncidentFeed_incidentFeed_edges_node } from "__generated__/GetIncidentFeed";
import { GetIncidentDetails, GetIncidentDetailsVariables } from "__generated__/GetIncidentDetails";

interface Props {
  incident: GetIncidentFeed_incidentFeed_edges_node;
  active: boolean;
  onClick: () => void;
}

export function TableRow({ incident, active, onClick }: Props) {
  const [loaded, setLoaded] = useState(false);
  const [, setLoadVideo] = useState(false);

  const seen = useMemo(() => {
    return active || SeenStateManager.isSeen(incident.uuid);
  }, [incident, active]);

  // TODO: this useEffect appears to be dead code...confirm and remove it
  useEffect(() => {
    // Remember if this row has been active/loaded before
    if (!loaded && active) {
      setLoaded(true);
    }

    // Unload the video player when no longer active
    if (loaded && !active) {
      setLoadVideo(false);
    }
  }, [active, loaded]);

  const thumbnailWrapperStyle = {
    backgroundImage: `url('${incident.camera?.thumbnailUrl}')`,
  };
  // incident.camera?.thumbnailUrl || undefined;
  return (
    <div className="border-t first:border-t-0 hover:bg-gray-50">
      <div className={classNames("flex gap-4 p-4 cursor-pointer", { "opacity-70": seen && !active })} onClick={onClick}>
        <div className="flex gap-4">
          <div
            className="w-24 h-16 bg-black bg-center bg-cover rounded-md overflow-hidden"
            style={thumbnailWrapperStyle}
          >
            {/* <img className="w-24" src={thumbnailUrl} alt={incident.incidentType?.name!} /> */}
          </div>
        </div>
        <div className="flex-grow">
          <div className="flex flex-col gap-1">
            <div className="font-bold">{incident.incidentType?.name || "Unknown"}</div>
            {incident.timestamp ? (
              <div>
                <DateTimeString dateTime={incident.timestamp} includeTimezone />
              </div>
            ) : null}
            <div>
              <StatusActionPill priority={incident.priority || null} status={incident.status || null} />
            </div>
          </div>
        </div>
        <div className="flex">
          <Bookmark incidentId={incident.id} bookmarked={incident.bookmarked} />
        </div>
      </div>
      <Transition
        show={active}
        enter="transition ease duration-300 transform"
        enterFrom="-translate-y-2"
        enterTo="translate-y-0"
      >
        <IncidentDetails incidentUuid={incident.uuid!} />
      </Transition>
    </div>
  );
}

function IncidentDetails(props: { incidentUuid: string }) {
  const { loading, data } = useQuery<GetIncidentDetails, GetIncidentDetailsVariables>(GET_INCIDENT_DETAILS, {
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
        <div className="flex gap-4 p-4 pt-0 flex-col lg:flex-row overflow-hidden">
          <div className="flex-grow">
            <Player
              videoUrl={data.incidentDetails.videoUrl!}
              annotationsUrl={data.incidentDetails.annotationsUrl}
              actorIds={data.incidentDetails.actorIds}
              annotationColorHex={BrightYellow}
              controls
            />
          </div>
          <div className="flex-grow-0">
            <div className="flex flex-col gap-2">
              <FeedbackButton
                incidentId={data.incidentDetails.id}
                incidentTitle={data.incidentDetails.incidentType?.name!}
              />
              <Button
                id={data.incidentDetails.id}
                data-ui-key="button-view-incident-details"
                variant="outlined"
                component={Link}
                to={`/incidents/${data.incidentDetails.uuid}`}
                startIcon={<ArrowSquareOut />}
              >
                View Details
              </Button>
              <AssignButton
                incidentId={data.incidentDetails.id}
                incidentUuid={data.incidentDetails.uuid}
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
