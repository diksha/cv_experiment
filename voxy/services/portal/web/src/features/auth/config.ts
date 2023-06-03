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
// trunk-ignore-all(gitleaks/generic-api-key): no secret keys here
import { Environment } from "shared/utilities/environment";

type Auth0Config = {
  domain: string;
  clientId: string;
  audience: string;
  redirectUri: string;
};

export const getAuth0Config = (environment: Environment): Auth0Config => {
  const redirectUri = `${window.location.origin}/auth/callback`;
  switch (environment) {
    case Environment.Production:
      return {
        domain: "auth.voxelai.com",
        clientId: "8JHNebSBU5NCQ0jW8CyqL0yac6KIqMWQ",
        audience:
          window.location.hostname === "app.production.voxelplatform.com"
            ? "https://app.production.voxelplatform.com/"
            : "https://app.voxelai.com/",
        redirectUri,
      };
    case Environment.Staging:
      return {
        domain: "voxelstaging.us.auth0.com",
        clientId: "OjhcDqaTZHq4onjtlryltMyCRzIqMZSz",
        audience:
          window.location.hostname === "app.staging.voxelplatform.com"
            ? "https://app.staging.voxelplatform.com/"
            : "https://app.staging.voxelai.com/",
        redirectUri,
      };
    case Environment.Development:
      return {
        domain: "voxeldev.us.auth0.com",
        clientId: "EKOwJD40kZerqjlxCIMETI5d4oeOb3cq",
        audience: "http://localhost:9000",
        redirectUri,
      };
    default:
      throw new Error(`Invalid environment for Auth0 config: ${environment}.`);
  }
};
