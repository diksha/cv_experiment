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
import React, { useCallback, useState, useEffect } from "react";
import { Link, useMatch } from "react-router-dom";
import { Slideover } from "ui";
import classNames from "classnames";
import { DrawerHeader } from "features/toolbox";
import { Wrench, X } from "phosphor-react";
import { CurrentOrganization, Scenarios, ClearSeenState, BetaFeatures } from "features/toolbox";
import { SiDjango, SiGraphql } from "react-icons/si";
import { ANALYTICS_PATH } from "features/analytics";
import { INCIDENT_FEED_PATH, INCIDENT_DETAILS_PATH } from "features/incidents";
import { REVIEW_HISTORY_PATH } from "features/mission-control/review";

const FORWARD_SLASH = 191;

const footerButtonClasses = classNames(
  "gird content-center p-3 rounded-full shadow-sm text-sm font-medium text-white",
  "hover:bg-white hover:bg-opacity-10",
  "focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
);

export function Toolbox() {
  const [open, setOpen] = useState(false);
  const incidentFeedPage = useMatch({
    path: INCIDENT_FEED_PATH,
  });
  const analyticsPage = useMatch(ANALYTICS_PATH);
  const incidentDetailsPage = useMatch(INCIDENT_DETAILS_PATH);
  const reviewHistoryPage = useMatch(REVIEW_HISTORY_PATH);
  const reposition = incidentFeedPage || reviewHistoryPage || analyticsPage;

  const keyboardShortcut = useCallback(
    (event) => {
      if (event.keyCode === FORWARD_SLASH) {
        setOpen(!open);
      }
    },
    [open]
  );

  const closeToolbox = () => {
    setOpen(false);
  };

  useEffect(() => {
    document.addEventListener("keydown", keyboardShortcut, false);

    return () => {
      document.removeEventListener("keydown", keyboardShortcut, false);
    };
  }, [keyboardShortcut]);

  return (
    <div
      className={classNames(
        "fixed bottom-16 md:bottom-4 right-4 md:right-4",
        "rounded-full bg-brand-gray-500 text-white",
        "shadow-lg hover:shadow-2xl",
        "opacity-20 hover:opacity-100 transition-opacity",
        // HACK: reposition this widget on pages with other floating widgets
        reposition ? "right-32" : ""
      )}
    >
      <button type="button" className="p-2" onClick={() => setOpen(!open)}>
        <Wrench className="h-6 w-6" />
      </button>
      <Slideover
        open={open}
        onClose={() => setOpen(false)}
        title={<DrawerHeader title="Toolbox" onClose={() => setOpen(false)} />}
        hideCloseButton
        dark
      >
        <div className="flex-grow text-white">
          {!!incidentDetailsPage?.params["incidentUuid"] ? (
            <Scenarios incidentUuid={incidentDetailsPage.params["incidentUuid"]} />
          ) : null}
          <CurrentOrganization />
          <BetaFeatures onFeatureClicked={closeToolbox} />
          <ClearSeenState />
        </div>
        <Slideover.Footer noPadding>
          <div className="flex gap-2 justify-between">
            <div className="flex">
              <Link target="_blank" to="/admin/" title="Django Admin" className={footerButtonClasses}>
                <SiDjango className="h-8 w-8" />
              </Link>
              <Link target="_blank" to="/graphql/" title="GraphQL IDE" className={footerButtonClasses}>
                <SiGraphql className="h-8 w-8" />
              </Link>
            </div>
            <button type="button" className={footerButtonClasses} onClick={() => setOpen(false)}>
              <X size={32} />
            </button>
          </div>
        </Slideover.Footer>
      </Slideover>
    </div>
  );
}
