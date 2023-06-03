import { GET_INCIDENT_FEEDBACK_DETAIL } from "features/incidents";
import { Link } from "react-router-dom";
import { Player } from "features/video";
/*
 * Copyright 2020-2021 Voxel Labs, Inc.
 * All rights reserved.
 *
 * This document may not be reproduced, republished, distributed, transmitted,
 * displayed, broadcast or otherwise exploited in any manner without the express
 * prior written permission of Voxel Labs, Inc. The receipt or possession of this
 * document does not convey any rights to reproduce, disclose, or distribute its
 * contents, or to manufacture, use, or sell anything that it may describe, in
 * whole or in part.
 */
import { Spinner } from "ui";
import { useQuery } from "@apollo/client";
import {
  GetIncidentFeedbackDetails,
  GetIncidentFeedbackDetailsVariables,
  GetIncidentFeedbackDetails_incidentDetails_feedback,
} from "__generated__/GetIncidentFeedbackDetails";

/**
 * Standalone minimal incident video player.
 */
export function IncidentVideo(props: { incidentId: string; incidentUuid: string }) {
  const { loading, data } = useQuery<GetIncidentFeedbackDetails, GetIncidentFeedbackDetailsVariables>(
    GET_INCIDENT_FEEDBACK_DETAIL,
    {
      variables: { incidentId: props.incidentId },
    }
  );

  const comments = data?.incidentDetails?.feedback.map(
    (item: GetIncidentFeedbackDetails_incidentDetails_feedback): JSX.Element => {
      return (
        <div key={item.id} className="py-1">
          <p className="font-bold">{item.user.email}</p>
          <p>{item.feedbackText || "No comment"}</p>
        </div>
      );
    }
  );

  return (
    <div>
      {loading ? (
        <div className="grid justify-center opacity-40">
          <Spinner />
        </div>
      ) : null}
      {!loading && data?.incidentDetails && (
        <div className="pt-0 pr-4 pb-4 pl-4 grid md:grid-cols-3 gap-4 sm:grid-cols-1">
          <div className="col-span-2">
            <Player
              videoUrl={data.incidentDetails.videoUrl!}
              annotationsUrl={data.incidentDetails.annotationsUrl}
              actorIds={data.incidentDetails.actorIds}
              controls
              cameraConfig={data.incidentDetails.cameraConfig}
            />
          </div>
          <div>
            <div className="font-bold">{data.incidentDetails.title}</div>
            <div>{comments}</div>
            <Link to={`/incidents/${props.incidentUuid}`} className="text-indigo-600 hover:text-indigo-900">
              View incident
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
