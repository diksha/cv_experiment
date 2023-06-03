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

// The term "profile" refers to portal-specific attributes of the current user
// vs. auth-specific attributes which come from Auth0.
export const GET_CURRENT_USER_PROFILE = gql`
  query GetCurrentUserProfile {
    currentUser {
      id
      roles {
        id
        name
        key
      }
      organization {
        id
        name
        key
        roles {
          id
          name
          key
        }
      }
      site {
        id
        name
        key
        timezone
        clientPreferences {
          key
          value
        }
      }
      sites {
        id
        key
        name
      }
      permissions
    }
  }
`;
