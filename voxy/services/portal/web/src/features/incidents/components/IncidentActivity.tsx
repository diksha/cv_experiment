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
import React, { useState, useEffect } from "react";
import { useMutation, useQuery } from "@apollo/client";
import { DateTime } from "luxon";
import { GET_INCIDENT_COMMENTS } from "features/incidents";
import classNames from "classnames";
import { Spinner } from "ui";
import { LoadingButton } from "@mui/lab";
import { ChatDots } from "phosphor-react";
import {
  Activity,
  ActivityType,
  ActivityFeed,
  SystemActivity,
  CommentActivity,
  AssignActivity,
} from "features/activity";
import { CREATE_COMMENT } from "../mutations";
import { GetComments, GetCommentsVariables } from "__generated__/GetComments";
import { CreateComment, CreateCommentVariables } from "__generated__/CreateComment";

export function IncidentActivity(props: { incidentId: string; incidentUuid: string; incidentTimestamp: string }) {
  const [activities, setActivities] = useState<Activity[]>([]);
  const [newCommentFocused, setNewCommentFocused] = useState(false);
  const [commentText, setCommentText] = useState("");
  const [commentValid, setCommentValid] = useState(false);
  const [commentError, setCommentError] = useState("");

  const MAX_COMMENT_LENGTH = 1000;

  const { loading, data, refetch } = useQuery<GetComments, GetCommentsVariables>(GET_INCIDENT_COMMENTS, {
    variables: {
      incidentId: props.incidentId,
    },
  });

  const [createComment, { loading: createCommentLoading }] = useMutation<CreateComment, CreateCommentVariables>(
    CREATE_COMMENT,
    {
      onCompleted: () => {
        setCommentText("");
        setCommentValid(false);
        refetch();
      },
    }
  );

  useEffect(() => {
    const newActivities: Activity[] = [];
    newActivities.push(
      new SystemActivity("incidentTimestamp", props.incidentTimestamp, {
        message: "Incident detected",
        incidentUuid: props.incidentUuid,
      })
    );

    if (data?.comments) {
      data.comments
        .slice()
        .sort((a, b) => {
          const aValue = DateTime.fromISO(a?.createdAt || 0);
          const bValue = DateTime.fromISO(b?.createdAt || 0);

          if (aValue < bValue) return -1;
          else if (aValue > bValue) return 1;

          return 0;
        })
        .forEach((comment: any) => {
          switch (comment.activityType) {
            case ActivityType.Comment:
              newActivities.push(
                new CommentActivity(comment.id, comment.createdAt, {
                  text: comment.text,
                  owner: comment.owner,
                  incidentUuid: props.incidentUuid,
                })
              );
              break;
            case ActivityType.Assign:
              newActivities.push(
                new AssignActivity(comment.id, comment.createdAt, {
                  text: comment.text,
                  owner: comment.owner,
                  note: comment.note,
                  incidentUuid: props.incidentUuid,
                })
              );
              break;
          }
        });
    }

    setActivities(newActivities);
  }, [data, props.incidentUuid, props.incidentTimestamp]);

  const handleCommentChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setCommentError("");
    let commentValue = e.target.value;
    let valid = true;
    if (commentValue.length === 0) {
      valid = false;
    }
    if (commentValue.length > MAX_COMMENT_LENGTH) {
      valid = false;
      setCommentError("Comments are limited to 1,000 characters.");
    }
    setCommentValid(valid);
    setCommentText(commentValue);
  };

  const handleCreateComment = () => {
    if (!commentValid) return;

    createComment({
      variables: {
        incidentId: props.incidentId,
        text: commentText,
      },
    });
  };

  return (
    <div>
      <h2 className="text-xl uppercase text-brand-blue-900 pb-4">Activity</h2>
      {loading ? (
        <div className="flex justify-center p-8">
          <Spinner />
        </div>
      ) : null}
      {!loading && data ? (
        <>
          <div className="flow-root">
            <ActivityFeed loading={loading} activities={activities} />
            <ul className="-mb-8">
              <li key="createComment">
                <div className="relative pb-8">
                  <div className="relative flex items-start space-x-3">
                    <div>
                      <div className="relative px-1">
                        <div className="h-8 w-8 bg-gray-100 rounded-full ring-8 ring-white flex items-center justify-center">
                          <ChatDots size={18} className="text-gray-500" aria-hidden="true" />
                        </div>
                      </div>
                    </div>
                    <div className="min-w-0 flex-1 py-1.5">
                      <div className="text-sm text-gray-500">
                        <textarea
                          id="about"
                          name="about"
                          rows={3}
                          className={classNames(
                            "shadow-sm block w-full rounded-md",
                            "sm:text-sm border border-gray-300",
                            "focus:border-brand-blue-900 focus:ring-brand-blue-900 "
                          )}
                          value={commentText}
                          onChange={handleCommentChange}
                          onFocus={() => setNewCommentFocused(true)}
                          onBlur={() => setNewCommentFocused(false)}
                        />
                        {newCommentFocused || commentError ? (
                          <div className="flex gap-x-4">
                            <div className="flex-grow text-red-500">{commentError ? commentError : null}</div>
                            <div
                              className={classNames(
                                "flex justify-end",
                                commentError ? "text-red-500" : "text-gray-400 -mb-8"
                              )}
                            >
                              {commentText.length}/{MAX_COMMENT_LENGTH}
                            </div>
                          </div>
                        ) : null}
                        <LoadingButton
                          variant="outlined"
                          className="mt-4"
                          onClick={handleCreateComment}
                          disabled={!commentValid}
                          loading={createCommentLoading}
                        >
                          Add comment
                        </LoadingButton>
                      </div>
                    </div>
                  </div>
                </div>
              </li>
            </ul>
          </div>
        </>
      ) : null}
    </div>
  );
}
