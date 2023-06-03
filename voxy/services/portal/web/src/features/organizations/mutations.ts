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
import { gql } from "@apollo/client";

export const CURRENT_USER_ORGANIZATION_UPDATE = gql`
  mutation CurrentUserOrganizationUpdate($organizationId: ID!) {
    currentUserOrganizationUpdate(organizationId: $organizationId) {
      status
    }
  }
`;

export const CURRENT_USER_SITE_UPDATE = gql`
  mutation CurrentUserSiteUpdate($siteId: ID!) {
    currentUserSiteUpdate(siteId: $siteId) {
      status
      newSiteId
    }
  }
`;
