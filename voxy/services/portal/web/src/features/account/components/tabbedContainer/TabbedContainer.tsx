import { ReactNode } from "react";
import { AccountTab } from "./enums";
import { HelpOutlineOutlined, PersonOutline, PeopleOutlineOutlined } from "@mui/icons-material";
import { Link } from "react-router-dom";
import { TabConfig } from "./types";
import { Card, Box, Tabs, Tab } from "@mui/material";
import { useRouting } from "shared/hooks";

interface TabbedContainerProps {
  activeTab: AccountTab;
  children: ReactNode;
}
export function TabbedContainer({ activeTab, children }: TabbedContainerProps) {
  return (
    <Card>
      <Box sx={{ display: "flex", gap: 2 }}>
        <Tabs
          orientation="vertical"
          variant="scrollable"
          value={activeTab}
          sx={{
            flexShrink: 0,
            borderRight: 1,
            borderColor: "divider",
            textAlign: "left",
            paddingY: 2,
            backgroundColor: (theme) => theme.palette.grey[200],
          }}
        >
          {Object.values(ACCOUNT_TABS).map((tabConfig) => (
            <TabbedContainerTab key={tabConfig.id} tabConfig={tabConfig} active={activeTab === tabConfig.id} />
          ))}
        </Tabs>
        <Box sx={{ padding: 2, width: "100%" }}>{children}</Box>
      </Box>
    </Card>
  );
}

interface TabbedContainerTabProps {
  tabConfig: TabConfig;
  active: boolean;
}
function TabbedContainerTab({ tabConfig, active }: TabbedContainerTabProps) {
  const { newLocationState } = useRouting();
  const Icon = tabConfig.icon;
  return (
    <Tab
      value={tabConfig.id}
      component={Link}
      to={tabConfig.path}
      state={newLocationState}
      label={tabConfig.label}
      icon={<Icon />}
      iconPosition="start"
      disableRipple
      sx={{
        paddingRight: 4,
        justifyContent: "start",
        minHeight: 50,
        fontWeight: active ? "bold" : "normal",
      }}
    />
  );
}

const ACCOUNT_TABS: Record<AccountTab, TabConfig> = {
  [AccountTab.Profile]: {
    id: AccountTab.Profile,
    label: "My Profile",
    path: "/account/profile",
    icon: PersonOutline,
  },
  [AccountTab.Teammates]: {
    id: AccountTab.Teammates,
    label: "Teammates",
    path: "/account/teammates",
    icon: PeopleOutlineOutlined,
  },
  [AccountTab.Support]: {
    id: AccountTab.Support,
    label: "Support",
    path: "/support",
    icon: HelpOutlineOutlined,
  },
};
