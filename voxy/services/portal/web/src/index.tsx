import { BrowserRouter } from "react-router-dom";
import * as Sentry from "@sentry/react";
import ThemeCustomization from "themes";
import { Integrations } from "@sentry/tracing";
import React from "react";
import ReactDOM from "react-dom";
import { HelmetProvider } from "react-helmet-async";
import App from "./App";
import { Auth0Provider } from "@auth0/auth0-react";
import { Environment, getCurrentEnvironment } from "shared/utilities/environment";
import { AuthProvider, getAuth0Config } from "features/auth";
import { AuthorizedApolloProvider } from "shared/apollo";
import { ScrollToTop } from "ui";

// Tailwind styles
import "assets/css/tailwind.preflight.css";
import "assets/css/tailwind.base.css";
import "assets/css/tailwind.components.css";
import "assets/css/tailwind.utilities.css";

// App base styles
import "assets/css/app.base.css";

// Material UI theme/styles
import "assets/scss/style.scss";

const currentEnvironment = getCurrentEnvironment();
const sentryDSNs: Record<Environment, string | null> = {
  [Environment.Production]: "https://a8d6ef15a3c343c5bcf77155e6e1ccac@st.public.voxelplatform.com/4",
  [Environment.Internal]: "https://c34f300f7c4a4db6815cb4169d6b11f3@st.public.voxelplatform.com/5",
  [Environment.Staging]: "https://c34f300f7c4a4db6815cb4169d6b11f3@st.public.voxelplatform.com/5",
  [Environment.Development]: null,
};
const sentryDSN = sentryDSNs[currentEnvironment];

if (sentryDSN) {
  Sentry.init({
    dsn: sentryDSN,
    integrations: [new Integrations.BrowserTracing()],

    // Set tracesSampleRate to 1.0 to capture 100%
    // of transactions for performance monitoring.
    // We recommend adjusting this value in production
    tracesSampleRate: 0.1,
  });
}

const auth0Config = getAuth0Config(currentEnvironment);

ReactDOM.render(
  <React.StrictMode>
    <Auth0Provider
      audience={auth0Config.audience}
      domain={auth0Config.domain}
      clientId={auth0Config.clientId}
      redirectUri={auth0Config.redirectUri}
    >
      <HelmetProvider>
        <AuthorizedApolloProvider>
          <AuthProvider>
            <ThemeCustomization>
              <BrowserRouter>
                <ScrollToTop />
                <App />
              </BrowserRouter>
            </ThemeCustomization>
          </AuthProvider>
        </AuthorizedApolloProvider>
      </HelmetProvider>
    </Auth0Provider>
  </React.StrictMode>,
  document.getElementById("root")
);
