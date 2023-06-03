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
import { CURRENT_USER_SITE_UPDATE } from "features/organizations";
import { useCurrentUser } from "features/auth";
import { CurrentUserSiteUpdateVariables, CurrentUserSiteUpdate } from "__generated__/CurrentUserSiteUpdate";
import { useSearchParams } from "react-router-dom";
import { useMutation } from "@apollo/client";
import { BackgroundSpinner } from "ui";
import { GenericErrorPage } from "features/errors/pages/GenericErrorPage";

const REDIRECT_SITE_KEY = "REDIRECT_SITE_KEY";

enum State {
  LOADING,
  INVALID_SITE_ERROR,
  GENERIC_ERROR,
  READY,
}

interface RedirectToSiteProps {
  children: React.ReactNode;
}
export function RedirectToSite({ children }: RedirectToSiteProps) {
  const { currentUser } = useCurrentUser();
  const [redirectState, setRedirectState] = useState(State.LOADING);
  const [userUpdate, { client }] = useMutation<CurrentUserSiteUpdate, CurrentUserSiteUpdateVariables>(
    CURRENT_USER_SITE_UPDATE,
    {
      onCompleted: () => {
        client.resetStore();
      },
    }
  );
  const [searchParams, setSearchParams] = useSearchParams();

  const removeRedirectParam = useCallback(() => {
    const params = Object.fromEntries(searchParams);
    delete params[REDIRECT_SITE_KEY];
    setSearchParams(params, { replace: true });
  }, [setSearchParams, searchParams]);

  useEffect(() => {
    const siteKey = searchParams.get(REDIRECT_SITE_KEY);
    if (!siteKey) {
      // No redirect expected
      setRedirectState(State.READY);
      return;
    }

    const targetSite = currentUser?.sites?.find((site) => site?.key === siteKey);
    if (!targetSite || !targetSite.id) {
      // Either the user doesn't have access or this site doesn't exist
      setRedirectState(State.INVALID_SITE_ERROR);
      return;
    }

    if (currentUser?.site?.key === siteKey) {
      // Current site matches target site, no redirect necessary
      removeRedirectParam();
      setRedirectState(State.READY);
      return;
    }

    setRedirectState(State.LOADING);
    userUpdate({
      variables: {
        siteId: targetSite.id,
      },
    })
      .then(() => {
        removeRedirectParam();
        setRedirectState(State.READY);
      })
      .catch(() => {
        setRedirectState(State.GENERIC_ERROR);
      });
  }, [userUpdate, currentUser, searchParams, removeRedirectParam]);

  const error = [State.GENERIC_ERROR, State.INVALID_SITE_ERROR].includes(redirectState);

  if (error) {
    return (
      <>
        {redirectState === State.GENERIC_ERROR && (
          <GenericErrorPage
            title="Access Denied - Voxel"
            message="Oops, something went wrong while loading the page."
          />
        )}
        {redirectState === State.INVALID_SITE_ERROR && (
          <GenericErrorPage
            title="Access Denied - Voxel"
            message={`It looks like you don't have access to the requested site, or the site doesn't exist. Check the URL and try again.`}
          />
        )}
      </>
    );
  }

  if (redirectState === State.LOADING) {
    return <BackgroundSpinner />;
  }

  return <>{children}</>;
}
