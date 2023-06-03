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
import { useCallback, useMemo } from "react";
import { Slideover } from "ui";
import { DateTime } from "luxon";
import { X, ClockCounterClockwise } from "phosphor-react";
import { CurrentZoneActivityFeed } from "features/activity";
import { useLocalStorage } from "shared/hooks";
import { Box } from "@mui/material";
import { useDashboardContext } from "features/dashboard/hooks/dashboard";

export function ActivitySheet(props: { zoneId: string; latestActivityTimestamp: DateTime }) {
  const { zoneId, latestActivityTimestamp } = props;
  const [lastViewedActivityTimestamp, setLastViewedActivityTimestamp] = useLocalStorage(
    // Local storage key is scoped to the current zone
    `notification:${zoneId}:lastViewedActivityTimestamp"`,
    ""
  );
  const { feedSlideOpen, setFeedSlideOpen } = useDashboardContext();

  const showRedDot = useMemo(() => {
    // There is activity + activity feed has never been viewed
    if (latestActivityTimestamp.isValid && !lastViewedActivityTimestamp) {
      return true;
    }

    // There is activity + activity feed has been viewed before
    if (latestActivityTimestamp && lastViewedActivityTimestamp) {
      return latestActivityTimestamp > DateTime.fromISO(lastViewedActivityTimestamp);
    }

    // There is no activity
    return false;
  }, [latestActivityTimestamp, lastViewedActivityTimestamp]);

  const handleOpen = useCallback(() => {
    setFeedSlideOpen(true);
    if (latestActivityTimestamp) {
      setLastViewedActivityTimestamp(latestActivityTimestamp.toISO());
    }
  }, [latestActivityTimestamp, setFeedSlideOpen, setLastViewedActivityTimestamp]);

  const handleClose = useCallback(() => {
    setFeedSlideOpen(false);
  }, [setFeedSlideOpen]);

  return (
    <div>
      <button
        data-ui-key="button-activity-feed"
        className="relative p-2 bg-brand-primary-100 rounded-full"
        onClick={handleOpen}
      >
        <ClockCounterClockwise className="w-6 h-6 text-brand-gray-500" />
        {showRedDot && <div className="absolute top-0 right-0 h-3 w-3 rounded-full bg-brand-red-500"></div>}
      </button>
      <Slideover open={feedSlideOpen} onClose={handleClose} hideCloseButton={true} noPadding={true}>
        <div className="sticky top-0 flex items-center p-4 bg-brand-gray-000 border-b border-brand-gray-100 z-10">
          <div className="flex-grow text-xl font-semibold font-epilogue text-brand-gray-500">Recent activity</div>
          <div>
            <button className="p-2 hover:bg-brand-gray-050 rounded-full" onClick={handleClose}>
              <X className="h-6 w-6" />
            </button>
          </div>
        </div>
        <Box p={2}>
          <CurrentZoneActivityFeed />
        </Box>
      </Slideover>
    </div>
  );
}
