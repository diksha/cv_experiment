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
import {
  CommentActivityFeedItem,
  AssignActivityFeedItem,
  SystemActivityFeedItem,
  AssignActivity,
  SystemActivity,
} from "features/activity";
import { Activity, CommentActivity } from "features/activity";

export function ActivityFeed(props: { loading: boolean; activities?: Activity[] }) {
  const activities = props.activities || [];
  return (
    <div>
      {props.loading ? <div className="py-24"></div> : null}
      {!props.loading && activities.length > 0 ? (
        <div className="flow-root">
          <ul className="">
            {activities.map((activity: Activity) => (
              <li key={activity.id}>
                {!activity.type && null}
                {activity instanceof CommentActivity && <CommentActivityFeedItem activity={activity} />}
                {activity instanceof AssignActivity && <AssignActivityFeedItem activity={activity} />}
                {activity instanceof SystemActivity && <SystemActivityFeedItem activity={activity} />}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {!props.loading && activities.length === 0 ? (
        <div className="pb-6 text-sm text-gray-400">Looks like there hasn't been any activity</div>
      ) : null}
    </div>
  );
}
