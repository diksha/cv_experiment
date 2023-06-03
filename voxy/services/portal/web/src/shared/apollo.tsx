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
import { useAuth0 } from "@auth0/auth0-react";
import Cookies from "js-cookie";
import React from "react";
import { setContext } from "@apollo/client/link/context";
import { ApolloClient, InMemoryCache, ApolloProvider, ApolloLink, createHttpLink } from "@apollo/client";
import { relayStylePagination } from "@apollo/client/utilities";

const httpLink = createHttpLink({
  uri: "/graphql/",
});

interface AuthorizedApolloProviderProps {
  children: React.ReactNode;
}
export const AuthorizedApolloProvider = ({ children }: AuthorizedApolloProviderProps) => {
  const { getAccessTokenSilently } = useAuth0();
  const authMiddleware = setContext(async (_, { headers, ...context }) => {
    // Need to have a fallback mechanism in development since localhost can't be
    // skipped for consent (unsafe origin) -> try silent, if fail, use consent flow.
    const accessToken = await getAccessTokenSilently();
    const csrfToken = Cookies.get("csrftoken");

    return {
      headers: {
        ...headers,
        // Auth
        ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
        "X-CSRFTOKEN": csrfToken || "",
        // Localization
        "X-VOXEL-CLIENT-TIMEZONE": Intl.DateTimeFormat().resolvedOptions().timeZone,
        "X-VOXEL-CLIENT-TIMEZONE-OFFSET": new Date().getTimezoneOffset(),
      },
      ...context,
    };
  });

  const client = new ApolloClient({
    link: ApolloLink.from([authMiddleware, httpLink]),
    cache: new InMemoryCache({
      typePolicies: {
        // IMPORTANT NOTES
        //   - Type policies are listed here with title-case type name as the key.
        //   - Connection-type fields must be configured here in order for
        //     Apollo to automagically handle relay style pagination.
        Query: {
          fields: {
            incidentFeed: relayStylePagination(),
            incidentFeedbackSummary: relayStylePagination(),
          },
        },
        ZoneType: {
          fields: {
            recentComments: relayStylePagination(),
            incidentFeed: relayStylePagination(),
            // Need to specify args so that Apollo includes argument values in cache keys
            incidents: relayStylePagination(["startDate", "endDate", "startTimestamp", "endTimestamp", "filters"]),
            incidentAnalytics: relayStylePagination(),
          },
        },
        OrganizationType: {
          fields: {
            sessionCount: relayStylePagination(),
          },
        },
        ProductionLine: {
          fields: {
            incidents: relayStylePagination(),
          },
        },
      },
    }),
    credentials: "same-origin",
  });

  return <ApolloProvider client={client}>{children}</ApolloProvider>;
};
