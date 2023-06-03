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
import axios from "axios";
import Cookies from "js-cookie";

const baseHeaders = {
  Accept: "application/json",
};

function redirectAuthIssues(url: String) {
  const ignoredUrls = ["/login"];

  const ignoreThisUrl = ignoredUrls.some((ignored) => url.includes(ignored));

  return !ignoreThisUrl;
}

const defaultInterceptor = (response: any) => response;

const errorInterceptor = (error: any) => {
  if (error && error.response) {
    const { status } = error.response;
    const redirectToLogin = [401, 403].includes(status) && redirectAuthIssues(error.request.responseURL);

    if (redirectToLogin) {
      alert("Unauthorized: You are no longer logged in.");
      window.location.href = "/login";
    }
  }

  return Promise.reject(error);
};

const requestInterceptor = (config: any) => {
  const headers = Object.assign({}, baseHeaders);
  config.headers = Object.assign(headers, config.headers || {});

  if (config.url.includes("storage.googleapis.com")) {
    // Special treatment for GCS requests
    config.withCredentials = false;
  } else {
    // Special treatment for all other requests
    const csrfToken = Cookies.get("csrftoken");
    if (csrfToken) {
      config.headers["X-CSRFTOKEN"] = csrfToken;
    }

    // Client localization headers
    config.headers["X-VOXEL-CLIENT-TIMEZONE"] = Intl.DateTimeFormat().resolvedOptions().timeZone;
    config.headers["X-VOXEL-CLIENT-TIMEZONE-OFFSET"] = new Date().getTimezoneOffset();
  }

  return config;
};

export const api = (options = {}) => {
  const baseApi = axios.create({
    timeout: 60000,
    withCredentials: true,
  });

  baseApi.interceptors.request.use(requestInterceptor);
  baseApi.interceptors.response.use(defaultInterceptor, errorInterceptor);

  return baseApi;
};
