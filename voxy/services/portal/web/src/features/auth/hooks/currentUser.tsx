import React, { useEffect, useState, useContext, useCallback, createContext } from "react";
import { useAuth0, LogoutOptions } from "@auth0/auth0-react";
import { RedirectLoginResult } from "@auth0/auth0-spa-js";
import { Auth0User, CurrentUser, AuthError, GET_CURRENT_USER_PROFILE } from "features/auth";
import { useLazyQuery } from "@apollo/client";
import { GetCurrentUserProfile } from "__generated__/GetCurrentUserProfile";
import { analytics } from "shared/utilities/analytics";

const stub = (): never => {
  throw new Error("You forgot to wrap your component in <AuthProvider>.");
};

export type RedirectAppState = {
  target?: string;
};

export interface AuthContextInterface {
  isLoading: boolean;
  isAuthenticated: boolean;
  currentUser?: CurrentUser;
  error?: AuthError;
  logout: (options?: LogoutOptions) => void;
  loginWithRedirect: () => Promise<void>;
  handleRedirectCallback: () => Promise<RedirectLoginResult<RedirectAppState>>;
  refresh: () => Promise<void>;
}

const initialContext = {
  isLoading: true,
  isAuthenticated: false,
  currentUser: undefined,
  error: undefined,
  logout: stub,
  loginWithRedirect: stub,
  handleRedirectCallback: stub,
  refresh: stub,
};

const AuthContext = createContext<AuthContextInterface>(initialContext);

export const AuthProvider = (props: { children: React.ReactNode }) => {
  const {
    isAuthenticated,
    isLoading: auth0Loading,
    user: auth0User,
    error: auth0Error,
    logout: auth0logout,
    loginWithRedirect,
    handleRedirectCallback,
    getAccessTokenSilently,
  } = useAuth0<Auth0User>();
  const [currentUser, setCurrentUser] = useState<CurrentUser>();
  const [
    fetchCurrentUserProfile,
    { data: profileData, loading: profileLoading, error: profileError, called: fetchProfileCalled },
  ] = useLazyQuery<GetCurrentUserProfile>(GET_CURRENT_USER_PROFILE, {
    // TODO(troycarlson): optimize this cache policy...
    // For some reason, with the default cache policy of "cache-first", this query gets
    // refetched on nearly every page load. Using "no-cache" leads to what I *think* is
    // our desired behavior: load the current user profile on initial application load,
    // and only refetch when this query is explicitly refetched (not based on Apollo's opaque
    // cache behavior).
    // fetchPolicy: "no-cache",
  });

  const error = auth0Error || profileError;

  useEffect(() => {
    if (auth0User && !fetchProfileCalled) {
      // Auth0 flow is finished, we're ready to fetch the profile
      fetchCurrentUserProfile();
    }
  }, [auth0User, fetchProfileCalled, fetchCurrentUserProfile]);

  useEffect(() => {
    if (auth0User && profileData) {
      const permissions = profileData.currentUser?.permissions || [];
      const organization = profileData.currentUser?.organization || undefined;
      const site = profileData.currentUser?.site || undefined;
      const sites = profileData.currentUser?.sites || undefined;
      const roles = profileData.currentUser?.roles || [];
      setCurrentUser(new CurrentUser(auth0User, permissions, roles, organization, site, sites));
      analytics.init(auth0User.email, organization?.key || "not_set", site?.key || "not_set");
    }
  }, [auth0User, profileData]);

  let logout = useCallback(
    (options?: LogoutOptions | undefined) => {
      auth0logout(options);
      fetch("/admin/logout/");
      analytics.reset();
    },
    [auth0logout]
  );

  return (
    <AuthContext.Provider
      value={{
        isLoading: auth0Loading || profileLoading,
        isAuthenticated,
        currentUser,
        error,
        logout,
        loginWithRedirect: () =>
          loginWithRedirect({
            scope: "permissions",
            appState: {
              target: window.location.pathname + window.location.search,
            } as RedirectAppState,
          }),
        handleRedirectCallback,
        refresh: async () => {
          try {
            // Force fetch a new access token which also results
            // in the Auth0 user object being updated
            await getAccessTokenSilently({ ignoreCache: true });
          } catch (error) {
            if (error instanceof Error && error.message.includes("Mulitfactor authentication required")) {
              return;
            }

            throw error;
          }
        },
      }}
    >
      {props.children}
    </AuthContext.Provider>
  );
};

export const useCurrentUser = () => {
  return useContext(AuthContext);
};
