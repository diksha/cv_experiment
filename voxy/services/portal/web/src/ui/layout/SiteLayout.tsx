import { AuthenticatedLayout } from "./AuthenticatedLayout";
import { SiteNav } from "./SiteNav";
import { Permission } from "features/auth";

interface SiteLayoutProps {
  children: React.ReactNode;
  requiredPermission?: Permission;
  mobileNavOnly?: boolean;
  hideRoutes?: boolean;
  hideSiteSelector?: boolean;
}

export function SiteLayout({
  children,
  requiredPermission,
  mobileNavOnly,
  hideRoutes,
  hideSiteSelector,
}: SiteLayoutProps) {
  return (
    <AuthenticatedLayout
      contextualNav={<SiteNav mobileNavOnly={mobileNavOnly} hideRoutes={hideRoutes} />}
      requiredPermission={requiredPermission}
      hideSiteSelector={hideSiteSelector}
      mobileNavOnly={mobileNavOnly}
    >
      {children}
    </AuthenticatedLayout>
  );
}
