import { Routes, Route, useLocation } from "react-router-dom";
import { SiteLayout } from "ui";
import { HomePage } from "features/home";
import { DashboardPageV3 } from "features/dashboard";
import {
  INCIDENT_FEED_PATH,
  IncidentsPage,
  IncidentDetailsPage,
  INCIDENT_DETAILS_PATH,
  SharedIncidentsPage,
} from "features/incidents";
import { Error404Page } from "features/errors";
import {
  ReviewPage,
  REVIEW_PATH,
  HistoryPage,
  REVIEW_HISTORY_PATH,
  ExperimentsPage,
  REVIEW_EXPERIMENTS_PATH,
} from "features/mission-control/review";

import {
  PAGE_ACCOUNT_CAMERAS,
  PAGE_INCIDENTS,
  PAGE_INCIDENT_DETAILS,
  PAGE_ANALYTICS,
  PAGE_REVIEW_QUEUE,
  PAGE_DASHBOARD,
  PAGE_EXECUTIVE_DASHBOARD,
  PAGE_REVIEW_HISTORY,
  EXPERIMENTAL_INCIDENTS_READ,
  RegistrationPage,
  AuthCallbackPage,
  LogoutPage,
} from "features/auth";
import { AnalyticsPage, AnalyticsPageV2, ANALYTICS_PATH } from "features/analytics";
import { DebugPage } from "features/common";
import { ProfilePage, TeammatesPage, AccountMenuPage } from "features/account";
import { CamerasPage } from "features/cameras";
import { SupportPage } from "features/support";
import { Helmet } from "react-helmet-async";
import { analytics } from "shared/utilities/analytics";
import { useEffect } from "react";
import { ExecutiveDashboardPage } from "features/executive-dashboard/pages";
import { DashboardProvider } from "features/dashboard/hooks/dashboard";

export default function App() {
  const location = useLocation();

  // Track pageviews when router location changes:
  // https://posthog.com/tutorials/spa
  useEffect(() => {
    analytics.trackPageview();
  }, [location]);

  return (
    <>
      <Helmet>
        <title>Voxel</title>
      </Helmet>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/login" element={<HomePage />} />
        <Route path="/logout" element={<LogoutPage />} />
        <Route path="/register/:token" element={<RegistrationPage />} />
        <Route path="/share/:token" element={<SharedIncidentsPage />} />
        <Route path="/auth/callback" element={<AuthCallbackPage />} />
        <Route path="/debug" element={<DebugPage />} />
        <Route
          path="/dashboard"
          element={
            <SiteLayout requiredPermission={PAGE_DASHBOARD}>
              <DashboardProvider>
                <DashboardPageV3 />
              </DashboardProvider>
            </SiteLayout>
          }
        />
        <Route
          path="/executive-dashboard"
          element={
            <SiteLayout requiredPermission={PAGE_EXECUTIVE_DASHBOARD} mobileNavOnly hideRoutes>
              <ExecutiveDashboardPage />
            </SiteLayout>
          }
        />
        <Route
          path={INCIDENT_FEED_PATH}
          element={
            <SiteLayout requiredPermission={PAGE_INCIDENTS}>
              <IncidentsPage />
            </SiteLayout>
          }
        />
        <Route
          path={ANALYTICS_PATH}
          element={
            <SiteLayout requiredPermission={PAGE_ANALYTICS}>
              <AnalyticsPage />
            </SiteLayout>
          }
        />
        <Route
          path="/beta/analytics"
          element={
            <SiteLayout requiredPermission={PAGE_ANALYTICS}>
              <AnalyticsPageV2 />
            </SiteLayout>
          }
        />
        <Route
          path={INCIDENT_DETAILS_PATH}
          element={
            <SiteLayout requiredPermission={PAGE_INCIDENT_DETAILS} hideSiteSelector>
              <IncidentDetailsPage />
            </SiteLayout>
          }
        />

        <Route
          path={REVIEW_PATH}
          element={
            <SiteLayout requiredPermission={PAGE_REVIEW_QUEUE} mobileNavOnly hideSiteSelector>
              <ReviewPage />
            </SiteLayout>
          }
        />
        <Route
          path={REVIEW_HISTORY_PATH}
          element={
            <SiteLayout requiredPermission={PAGE_REVIEW_HISTORY} mobileNavOnly hideSiteSelector>
              <HistoryPage />
            </SiteLayout>
          }
        />
        <Route
          path={REVIEW_EXPERIMENTS_PATH}
          element={
            <SiteLayout requiredPermission={EXPERIMENTAL_INCIDENTS_READ} mobileNavOnly hideSiteSelector>
              <ExperimentsPage />
            </SiteLayout>
          }
        />
        <Route
          path="/account"
          element={
            <SiteLayout mobileNavOnly hideRoutes>
              <AccountMenuPage />
            </SiteLayout>
          }
        />
        <Route
          path="/account/profile"
          element={
            <SiteLayout mobileNavOnly hideRoutes>
              <ProfilePage />
            </SiteLayout>
          }
        />
        <Route
          path="/account/teammates"
          element={
            <SiteLayout mobileNavOnly hideRoutes>
              <TeammatesPage />
            </SiteLayout>
          }
        />
        <Route
          path="/cameras"
          element={
            <SiteLayout requiredPermission={PAGE_ACCOUNT_CAMERAS}>
              <CamerasPage />
            </SiteLayout>
          }
        />
        <Route
          path="/support"
          element={
            <SiteLayout mobileNavOnly hideRoutes>
              <SupportPage />
            </SiteLayout>
          }
        />
        {/* Catch-all */}
        <Route path="*" element={<Error404Page />} />
      </Routes>
    </>
  );
}
