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
import { useNavigate } from "react-router-dom";
import { useCurrentUser } from "features/auth";
import { GenericErrorPage } from "features/errors";
import { BackgroundSpinner } from "ui";

export function AuthCallbackPage() {
  const navigate = useNavigate();
  const [error, setError] = useState<unknown>();
  const { currentUser, handleRedirectCallback } = useCurrentUser();

  useEffect(() => {
    const handleCallback = async () => {
      try {
        const result = await handleRedirectCallback();
        // Redirect to the original URL the user was trying to load (or root if not specified)
        navigate(result.appState?.target || "/");
      } catch (error) {
        console.error(error);
        if (currentUser) {
          // If currentUser appears valid, try redirecting to the home page
          console.info("Session appears valid, redirecting to home page...");
          navigate("/");
        } else {
          // Otherwise display an error page
          setError(error);
        }
      }
    };

    handleCallback();
  }, [navigate, handleRedirectCallback, currentUser]);

  return error ? (
    <GenericErrorPage
      title="Something went wrong"
      message="Something went wrong while logging in, try clicking Go to Dashboard or contact support at support@voxelai.com."
    />
  ) : (
    <BackgroundSpinner />
  );
}
