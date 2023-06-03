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
import { Tag } from "phosphor-react";

export function IncidentTag(props: { label?: string; value: string }) {
  return (
    <div className="inline-block py-0.5 px-2 text-xs bg-white text-brand-gray-500 border border-brand-gray-100 rounded-full">
      <div className="flex gap-2 items-center">
        <Tag className="inline-block text-brand-gray-500" size="14px" weight="fill" />
        <div>{props.value}</div>
      </div>
    </div>
  );
}
