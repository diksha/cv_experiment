import { Card, Spinner } from "ui";
import { Button } from "@mui/material";
import { GET_INCIDENT_FEEDBACK_QUEUE, IncidentFeedback } from "features/incidents";
import React, { useState, useEffect } from "react";
import { TabHeader } from "../components";
import { Helmet } from "react-helmet-async";
import { Player } from "features/video";
import { Logtail } from "@logtail/browser";
import { useCurrentUser } from "features/auth";
import classNames from "classnames";
import pageStyles from "shared/styles/pages.module.css";
import styles from "./ReviewPage.module.css";
import { useLazyQuery } from "@apollo/client";
import { Environment, getCurrentEnvironment } from "shared/utilities/environment";
import {
  GetIncidentFeedbackQueue,
  GetIncidentFeedbackQueueVariables,
  GetIncidentFeedbackQueue_incidentFeedbackQueue as ifq,
} from "__generated__/GetIncidentFeedbackQueue";
export const REVIEW_PATH = "/review";
// The number of incident reviews to prefetch
export const REVIEW_BUFFER_SIZE = 3;
// The polling frequency to check buffer state after seeing empty buffer.
export const QUEUE_POLLING_INTERVAL_MS = 3 * 1000;
// trunk-ignore-all(gitleaks/generic-api-key)
const PROD_LOGTAIL_KEY = "wx1AFYSpUyf2gfQH2Kc1UK7i";
const DEV_LOGTAIL_KEY = "vxpWXtYjzVT33h6a6W1Fr2hc";
const REVIEW_EVENT_TABLE_NAME = "REVIEW_PAGE_EVENT_TABLE";

const currentEnvironment = getCurrentEnvironment();
const currentLogtailKey = currentEnvironment === Environment.Production ? PROD_LOGTAIL_KEY : DEV_LOGTAIL_KEY;
const logtail = new Logtail(currentLogtailKey);

export function ReviewPage(): JSX.Element {
  const { currentUser } = useCurrentUser();
  const [timestampOfMostRecentSubmit, setTimestampOfMostRecentSubmit] = useState<number>(Date.now());
  const [activePanelId, setActivePanelId] = useState<number>(0);
  const [incidentBuffer, setIncidentBuffer] = useState<string[]>([]);

  function logMeasurements(elapsedTimeBetweenSubmissions: number, incidentQueue: ifq | null) {
    const reviewer_email = currentUser?.email || "";
    const msg = {
      table: REVIEW_EVENT_TABLE_NAME,
      incident_uuid: incidentQueue?.uuid || "unknown",
      elapsed_time_submission: elapsedTimeBetweenSubmissions,
      incident_type_key: incidentQueue?.incidentType?.key || "unknown",
      reviewer_email: reviewer_email,
    };

    logtail.info("Review Page Event", msg);
  }

  function markReviewSubmission(currentTime: number, incidentQueue: ifq | null) {
    const elapsedTimeBetweenSubmissions = currentTime - timestampOfMostRecentSubmit;
    logMeasurements(elapsedTimeBetweenSubmissions, incidentQueue);

    setTimestampOfMostRecentSubmit(currentTime);
  }

  const updateIncidentBuffer = (reviewPanelId: number, incident_uuid: string) => {
    setIncidentBuffer((previous) => {
      const newBuffer = previous.slice();
      newBuffer[reviewPanelId] = incident_uuid;
      return newBuffer;
    });
  };

  const handleTransition = (currentTime: number, incidentQueue: ifq | null) => {
    markReviewSubmission(currentTime, incidentQueue);
    setActivePanelId((previous) => (previous + 1) % REVIEW_BUFFER_SIZE);
  };

  const reviewPanels = Array(REVIEW_BUFFER_SIZE)
    .fill(0)
    .map((_, index) => (
      <ReviewPagePanel
        key={index}
        reviewPanelId={index}
        activePanelId={activePanelId}
        onSubmit={handleTransition}
        incidentBuffer={incidentBuffer}
        updateIncidentBuffer={updateIncidentBuffer}
        timestampOfMostRecentSubmit={timestampOfMostRecentSubmit}
      />
    ));

  return (
    <>
      <Helmet>
        <title>Incident Review - Voxel</title>
      </Helmet>
      <TabHeader selectedTab={TabHeader.Tab.Review} />
      <div className={pageStyles.page}>{reviewPanels}</div>
    </>
  );
}

export function ReviewPagePanel(props: {
  reviewPanelId: number;
  activePanelId: number;
  onSubmit: (currentTime: number, incident: ifq | null) => void;
  incidentBuffer: string[];
  updateIncidentBuffer: (reviewPanelId: number, incident_uuid: string) => void;
  timestampOfMostRecentSubmit: number;
}): JSX.Element {
  const [isInitialized, setIsInitialized] = useState(false);
  const [showBoundingBox, setShowBoundingBox] = useState(true);

  const [getIncidentFeedbackQueue, { loading, data: incident, refetch, startPolling, stopPolling }] = useLazyQuery<
    GetIncidentFeedbackQueue,
    GetIncidentFeedbackQueueVariables
  >(GET_INCIDENT_FEEDBACK_QUEUE, {
    fetchPolicy: "no-cache",
    onCompleted: (data) => {
      if (data?.incidentFeedbackQueue?.[0]) {
        // TODO: Is it ok to call this consistently?
        stopPolling();
        props.updateIncidentBuffer(props.reviewPanelId, data.incidentFeedbackQueue[0].uuid);
      } else {
        setTimeout(
          () =>
            refetch({
              reviewQueueContext: {
                reviewPanelId: props.reviewPanelId,
                incidentExclusionList: props.incidentBuffer,
              },
            }),
          QUEUE_POLLING_INTERVAL_MS
        ); // Refill buffer if it's empty
      }
    },
  });

  useEffect(() => {
    if (!isInitialized && props.reviewPanelId <= props.incidentBuffer.length) {
      getIncidentFeedbackQueue({
        variables: {
          reviewQueueContext: {
            reviewPanelId: props.reviewPanelId,
            incidentExclusionList: props.incidentBuffer,
          },
        },
      });
      setIsInitialized(true);
    }
  }, [isInitialized, props.reviewPanelId, props.incidentBuffer, getIncidentFeedbackQueue]);

  const incidentFeedbackQueue = incident?.incidentFeedbackQueue!;
  const isIncidentQueueEmpty = incidentFeedbackQueue && incidentFeedbackQueue.length === 0;
  const ready = Boolean(!loading && incident);
  const incidentQueue = ready && !isIncidentQueueEmpty ? incidentFeedbackQueue[0] : null;

  useEffect(() => {
    /**
     * If the queue is empty:
     * 1. Reload the page (resets activePanel to 0 and clears buffers)
     * 2. After reload: start polling periodically
     * 3. After first poll success (onCompleted) -> stop polling.
     */
    if (props.reviewPanelId === props.activePanelId && !loading && incident && isIncidentQueueEmpty) {
      // Ensures polling can/only begin after fresh reload
      if (props.reviewPanelId === 0 && props.incidentBuffer.length === 0) {
        startPolling(QUEUE_POLLING_INTERVAL_MS);
      } else {
        window.location.reload();
      }
    }
  }, [
    props.reviewPanelId,
    props.activePanelId,
    props.incidentBuffer,
    loading,
    incident,
    isIncidentQueueEmpty,
    startPolling,
  ]);

  /**
   * Handles transitions between queue items after feedback is submitted.
   */
  const handleTransition = () => {
    props.onSubmit(Date.now(), incidentQueue);
    refetch({
      reviewQueueContext: {
        reviewPanelId: props.reviewPanelId,
        incidentExclusionList: props.incidentBuffer,
      },
    });
    setShowBoundingBox(true);
  };

  return (
    <div hidden={props.activePanelId !== props.reviewPanelId}>
      <Card>
        {loading && (
          <div className={styles.spinner}>
            <Spinner />
          </div>
        )}
        {!loading && incident && (
          <div className={classNames(styles.contentWrapper, "grid grid-cols-1 lg:grid-cols-2")}>
            {isIncidentQueueEmpty ? (
              <div>
                There are no incidents available for review at this time. Automatically refreshing every{" "}
                {QUEUE_POLLING_INTERVAL_MS / 1000} seconds...{" "}
              </div>
            ) : (
              <>
                <div>
                  {ready && (
                    <>
                      <Player
                        key={`video-${incidentQueue!.id}`}
                        videoUrl={incidentQueue!.videoUrl!}
                        annotationsUrl={incidentQueue!.annotationsUrl!}
                        actorIds={incidentQueue?.actorIds}
                        controls
                        cameraConfig={incidentQueue!.cameraConfig}
                        hideCanvas={!showBoundingBox}
                        autoplay={props.activePanelId === props.reviewPanelId}
                      />
                      <div className="py-2">Incident UUID: {incidentQueue!.uuid}</div>
                    </>
                  )}
                  <div className="py-4">
                    {ready && (
                      <Button variant="contained" onClick={() => setShowBoundingBox(!showBoundingBox)}>
                        {showBoundingBox ? "Hide" : "Show"} bounding box
                      </Button>
                    )}
                  </div>
                </div>
                {ready && (
                  <IncidentFeedback
                    key={`feedback-${incidentQueue!.id!}`}
                    incidentId={incidentQueue!.id!}
                    incidentTitle={incidentQueue!.title!}
                    incidentOrganization={`${incidentQueue!.organization!.name!} ${incidentQueue!.zone!.name!}`}
                    onSubmit={handleTransition}
                    timestampOfMostRecentSubmit={props.timestampOfMostRecentSubmit}
                  />
                )}
              </>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
