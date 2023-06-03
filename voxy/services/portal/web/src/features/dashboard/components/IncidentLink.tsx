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
import { Link } from "react-router-dom";
import { toTitleCase } from "shared/utilities/strings";
import { CaretRight } from "phosphor-react";
import { DateTime } from "luxon";

export function IncidentLink(props: { incidentUuid?: string; title?: string; timestamp?: string }) {
  return (
    <Link
      className="block border-t py-2 pl-4 pr-2 hover:bg-gray-100"
      key={`incidentLink-${props.incidentUuid}`}
      to={`/incidents/${props.incidentUuid}`}
    >
      <div className="flex">
        <div className="flex-grow">
          <div>{toTitleCase(props.title || "Unknown")}</div>
          {props.timestamp ? (
            <div className="text-sm text-gray-400">{DateTime.fromISO(props.timestamp).toRelative()}</div>
          ) : null}
        </div>
        <div className="grid items-center">
          <CaretRight size={18} className="text-gray-400" />
        </div>
      </div>
    </Link>
  );
}
