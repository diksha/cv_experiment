/*
 * Copyright 2020-2022 Voxel Labs, Inc.
 * All rights reserved.
 *
 * This document may not be reproduced, republished, distributed, transmitted,
 * displayed, broadcast or otherwise exploited in any manner without the express
 * prior written permission of Voxel Labs, Inc. The receipt or possession of this
 * document does not convey any rights to reproduce, disclose, or distribute its
 * contents, or to manufacture, use, or sell anything that it may describe, in
 * whole or in part.
 */
import React, { useState } from "react";
import classNames from "classnames";
import { ArrowsClockwise } from "phosphor-react";
import { useInterval } from "shared/hooks";
import { DateTime } from "luxon";

const AUTO_REFRESH_INTERVAL_SECONDS = 120;
const AUTO_REFRESH_INTERVAL_MILLISECONDS = AUTO_REFRESH_INTERVAL_SECONDS * 1000;

export function AutoRefresh(props: {
  lastUpdatedTimestamp?: DateTime;
  loading: boolean;
  refreshing: boolean;
  handleRefresh: () => void;
}) {
  const { lastUpdatedTimestamp, loading, refreshing, handleRefresh } = props;
  const [lastUpdatedTimestampMessage, setLastUpdatedTimestampMessage] = useState<string | null>("Loading...");
  useInterval(() => {
    setLastUpdatedTimestampMessage(`Last updated ${lastUpdatedTimestamp?.toRelative()}.`);
  }, 1000);
  useInterval(() => {
    handleRefresh();
  }, AUTO_REFRESH_INTERVAL_MILLISECONDS);
  return (
    <>
      <div className={classNames("transition-opacity", loading || refreshing ? "opacity-0" : "opacity-100 delay-1000")}>
        {lastUpdatedTimestampMessage}
      </div>
      <button className="flex gap-2 py-1 px-2 rounded-md items-center hover:bg-brand-gray-050" onClick={handleRefresh}>
        <span>Refresh</span>
        <ArrowsClockwise className={classNames("h-5 w-5", { "animate-spin": refreshing })} />
      </button>
    </>
  );
}
