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

/**
 * Used for constistent right/left padding for page-level content.
 */
export function ContentWrapper(props: { children: React.ReactNode }) {
  return <div className="px-0 md:px-4 max-w-screen-xl">{props.children}</div>;
}
