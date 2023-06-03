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
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useCurrentUser } from "features/auth";
import { AccountErrorPage } from "features/errors";
import { BackgroundSpinner } from "ui";
import { PAGE_DASHBOARD, PAGE_EXECUTIVE_DASHBOARD, PAGE_REVIEW_QUEUE } from "features/auth";

export function HomePage() {
  const navigate = useNavigate();
  const [accountError, setAccountError] = useState(false);
  const { currentUser, isLoading, isAuthenticated, loginWithRedirect } = useCurrentUser();

  useEffect(() => {
    if (!isLoading) {
      if (currentUser?.hasGlobalPermission(PAGE_EXECUTIVE_DASHBOARD)) {
        navigate("/executive-dashboard", { replace: true });
      } else if (currentUser?.hasGlobalPermission(PAGE_DASHBOARD)) {
        navigate("/dashboard", { replace: true });
      } else if (currentUser?.hasGlobalPermission(PAGE_REVIEW_QUEUE)) {
        navigate("/review", { replace: true });
      } else if (!isAuthenticated) {
        loginWithRedirect();
      } else if (currentUser) {
        setAccountError(true);
      }
    }
  }, [navigate, currentUser, isLoading, isAuthenticated, loginWithRedirect]);

  return accountError ? <AccountErrorPage /> : <BackgroundSpinner />;
}
