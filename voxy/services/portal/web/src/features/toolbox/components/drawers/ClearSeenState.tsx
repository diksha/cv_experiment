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
import React, { useEffect, useState } from "react";
import { Spinner } from "ui";
import { Drawer } from "features/toolbox";
import { SeenStateManager } from "features/incidents";
import { useNavigate } from "react-router-dom";
import { EyeSlash } from "phosphor-react";

function Content() {
  const navigate = useNavigate();
  const [cleared, setCleared] = useState(false);

  useEffect(() => {
    SeenStateManager.clear();
    setCleared(true);
  }, []);

  useEffect(() => {
    if (cleared) {
      // Refresh the page
      navigate(0);
    }
  }, [cleared, navigate]);

  return (
    <div className="w-full p-4 text-center">
      <Spinner white />
    </div>
  );
}

export function ClearSeenState() {
  return (
    <Drawer name="Clear seen state" icon={<EyeSlash className="h-6 w-6 text-brand-blue-300" />}>
      <Content />
    </Drawer>
  );
}
