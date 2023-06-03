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
import React from "react";
import { Warning, ChatDots, UserPlus } from "phosphor-react";
import { SystemActivity, AssignActivity, CommentActivity } from "features/activity";
import { useCurrentUser, User } from "features/auth";
import { DateTime } from "luxon";
import { Avatar, formatDateTimeString } from "ui";
import { Link, useLocation } from "react-router-dom";
import classNames from "classnames";

function ActivityFeedItem(props: {
  user?: User;
  title: string;
  timestamp: string;
  timestampPrefix?: string;
  icon: React.ReactNode;
  children?: React.ReactNode;
  incidentUuid?: string;
}) {
  const location = useLocation();
  const name = props.user?.fullName;
  const relativeTimestamp = DateTime.fromISO(props.timestamp).toRelative();
  const prefix = props.timestampPrefix || "";
  const timestampString = `${prefix} ${relativeTimestamp}`.trim();

  const incidentLinkPath = `/incidents/${props.incidentUuid}`;
  const renderIncidentLinks = !location.pathname.includes(incidentLinkPath);

  const { currentUser } = useCurrentUser();

  const content = (
    <div className="group relative pb-8">
      <span className="absolute top-5 left-5 h-full w-0.5 bg-gray-200" aria-hidden="true" />
      <div className="relative flex items-start space-x-3">
        <div className="relative">
          {props.user ? (
            <>
              <Avatar url={props.user?.picture} name={name} />
              <span className="absolute -bottom-0.5 -right-1 text-gray-400 bg-white rounded-tl px-0.5 py-px">
                {props.icon}
              </span>
            </>
          ) : (
            <div className="h-10 w-10 mx-px mr-1 bg-gray-100 text-gray-500 rounded-full ring-8 ring-white flex items-center justify-center">
              {props.icon}
            </div>
          )}
        </div>
        <div className="min-w-0 flex-1">
          <div>
            <div className="text-sm">
              <div className="font-medium text-gray-900">{props.title}</div>
            </div>
            <p
              className="mt-0.5 text-sm text-gray-500"
              title={formatDateTimeString(props.timestamp, currentUser?.site?.timezone)}
            >
              <span className={classNames(renderIncidentLinks && "group-hover:underline")}>{timestampString}</span>
            </p>
          </div>
          {props.children ? <div className="mt-2 text-sm text-gray-700">{props.children}</div> : null}
        </div>
      </div>
    </div>
  );

  return renderIncidentLinks ? <Link to={incidentLinkPath}>{content}</Link> : <>{content}</>;
}

export function CommentActivityFeedItem(props: { activity: CommentActivity }) {
  const { timestamp, data } = props.activity;
  const { owner, text, incidentUuid } = data;

  return (
    <ActivityFeedItem
      user={owner}
      timestamp={timestamp}
      timestampPrefix="Commented"
      title={owner.fullName}
      icon={<ChatDots />}
      incidentUuid={incidentUuid}
    >
      <p>{text}</p>
    </ActivityFeedItem>
  );
}

export function SystemActivityFeedItem(props: { activity: SystemActivity }) {
  const { timestamp, data } = props.activity;
  const { message, incidentUuid } = data;

  return (
    <ActivityFeedItem timestamp={timestamp} title={message} icon={<Warning size={24} />} incidentUuid={incidentUuid} />
  );
}

export function AssignActivityFeedItem(props: { activity: AssignActivity }) {
  const { timestamp, data } = props.activity;
  const { owner, text, incidentUuid, note } = data;

  return (
    <ActivityFeedItem user={owner} timestamp={timestamp} title={text} icon={<UserPlus />} incidentUuid={incidentUuid}>
      <p>{note}</p>
    </ActivityFeedItem>
  );
}
