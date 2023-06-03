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

const PRODUCTION_DOMAIN = "app.voxelai.com";
const PRODUCTION_PLATFORM_DOMAIN = "app.production.voxelplatform.com";
const INTERNAL_DOMAIN = "app.internal.voxelai.com";
const STAGING_DOMAIN = "app.staging.voxelai.com";
const STAGING_PLATFORM_DOMAIN = "app.staging.voxelplatform.com";
const DEVELOPMENT_DOMAIN = "localhost";

export enum Environment {
  Development = "DEVELOPMENT",
  Internal = "INTERNAL",
  Staging = "STAGING",
  Production = "PRODUCTION",
}

export const getCurrentEnvironment = () => {
  switch (window.location.hostname) {
    case PRODUCTION_DOMAIN:
    case PRODUCTION_PLATFORM_DOMAIN:
      return Environment.Production;
    case INTERNAL_DOMAIN:
      return Environment.Internal;
    case STAGING_DOMAIN:
    case STAGING_PLATFORM_DOMAIN:
      return Environment.Staging;
    case DEVELOPMENT_DOMAIN:
      return Environment.Development;
    default:
      throw new Error(`No environment defined for hostname: ${window.location.hostname}.`);
  }
};
