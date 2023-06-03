/*
 * Copyright 2022 Voxel Labs, Inc.
 * All rights reserved.
 *
 * This document may not be reproduced, republished, distributed, transmitted,
 * displayed, broadcast or otherwise exploited in any manner without the express
 * prior written permission of Voxel Labs, Inc. The receipt or possession of this
 * document does not convey any rights to reproduce, disclose, or distribute its
 * contents, or to manufacture, use, or sell anything that it may describe, in
 * whole or in part.
 */
import { DateTime } from "luxon";
import { EmptyRangeFeedItemData } from "features/incidents";

export function EmptyRangeFeedItem(props: { data: EmptyRangeFeedItemData }) {
  const startDate = DateTime.fromISO(props.data.startDate).toFormat("MMMM d");
  const endDate = DateTime.fromISO(props.data.endDate).toFormat("MMMM d");

  let message;
  if (startDate === endDate) {
    message = `No activity on ${startDate}`;
  } else {
    message = `No activity from ${startDate} to ${endDate}`;
  }
  return <div className="text-center text-sm text-gray-400 font-bold">{message}</div>;
}
