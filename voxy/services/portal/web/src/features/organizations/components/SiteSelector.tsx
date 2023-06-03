import { useQuery, useMutation } from "@apollo/client";
import { useEffect, useMemo, useState } from "react";
import { GetCurrentSiteData, GetCurrentSiteData_currentUser_sites } from "__generated__/GetCurrentSiteData";
import { CurrentUserSiteUpdateVariables, CurrentUserSiteUpdate } from "__generated__/CurrentUserSiteUpdate";
import { GET_CURRENT_SITE_DATA, CURRENT_USER_SITE_UPDATE } from "features/organizations";
import { WarehouseOutlined } from "@mui/icons-material";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { filterNullValues } from "shared/utilities/types";
import { Box, ListSubheader, Skeleton, FormControl, SelectChangeEvent, Select, MenuItem } from "@mui/material";
import { PAGE_EXECUTIVE_DASHBOARD, useCurrentUser } from "features/auth";

interface SiteOption extends GetCurrentSiteData_currentUser_sites {}

const EXECUTIVE_DASHBOARD = "EXECUTIVE_DASHBOARD";
const allSitesOption: SiteOption = {
  __typename: "ZoneType",
  id: EXECUTIVE_DASHBOARD,
  key: EXECUTIVE_DASHBOARD,
  name: "All Sites",
  isActive: true,
};

export function SiteSelector() {
  const { currentUser } = useCurrentUser();
  const navigate = useNavigate();
  const location = useLocation();
  let [, setSearchParams] = useSearchParams();
  const [selected, setSelected] = useState<SiteOption>();
  const { loading: queryLoading, data } = useQuery<GetCurrentSiteData>(GET_CURRENT_SITE_DATA);
  const [userUpdate, { loading: mutationLoading }] = useMutation<CurrentUserSiteUpdate, CurrentUserSiteUpdateVariables>(
    CURRENT_USER_SITE_UPDATE
  );

  const allSites = useMemo(() => {
    return filterNullValues<SiteOption>(data?.currentUser?.sites).filter((site) => site.id && site.name);
  }, [data]);

  const activeSites = useMemo(() => {
    const result = allSites.filter((site) => site.isActive);
    if (currentUser?.hasGlobalPermission(PAGE_EXECUTIVE_DASHBOARD)) {
      result.unshift(allSitesOption);
    }
    return result;
  }, [allSites, currentUser]);

  useEffect(() => {
    if (location.pathname === "/executive-dashboard") {
      setSelected(allSitesOption);
    } else {
      setSelected(allSites.find((site) => site.id === data?.currentUser?.site?.id));
    }
  }, [allSites, data, location.pathname]);

  const handleChange = async (event: SelectChangeEvent<string>) => {
    const siteId = event.target.value;
    if (siteId === EXECUTIVE_DASHBOARD) {
      navigate("/executive-dashboard");
      return;
    }
    try {
      await userUpdate({
        variables: {
          siteId,
        },
      });

      if (location.pathname === "/executive-dashboard") {
        navigate("/dashboard", { replace: true });
      } else {
        setSearchParams("");
      }
      navigate(0);
    } catch {
      alert("Sorry, something went wrong while switching sites");
    }
  };

  if (queryLoading) {
    return <Skeleton variant="rounded" height="40px" width="160px" />;
  }

  const organizationName = data?.currentUser?.organization?.name;

  // TODO: handle case where only one site is available
  const renderValue = (siteId: string) => {
    const site = siteId === EXECUTIVE_DASHBOARD ? allSitesOption : allSites.find((site) => site.id === siteId);
    return (
      <Box sx={{ display: "flex", gap: 2, alignItems: "center" }}>
        <WarehouseOutlined />
        {organizationName} - {site?.name}
      </Box>
    );
  };

  if (!activeSites.length) {
    return <></>;
  }
  return (
    <FormControl>
      <Select
        id="current-site-select"
        value={selected?.id || ""}
        size="small"
        disabled={mutationLoading}
        onChange={handleChange}
        renderValue={renderValue}
      >
        <ListSubheader sx={{ fontWeight: "bold" }}>{organizationName}</ListSubheader>
        {activeSites.map((site) => (
          <MenuItem key={site.id} value={site.id} sx={{ paddingLeft: 4 }}>
            {site.name}
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
}
