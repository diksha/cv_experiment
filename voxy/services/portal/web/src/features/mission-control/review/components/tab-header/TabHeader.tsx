import { Link } from "react-router-dom";
import React from "react";
import { useTheme, Tabs, Tab, Box } from "@mui/material";
import { ReviewTab } from "./enums";
import { TabConfigs } from "./constants";
import { useCurrentUser } from "features/auth";

interface TabHeaderProps {
  selectedTab: ReviewTab;
}
export function TabHeader({ selectedTab }: TabHeaderProps) {
  const { currentUser } = useCurrentUser();
  const theme = useTheme();
  const tabStyles = {
    backgroundColor: theme.palette.background.paper,
    "& .MuiTabs-flexContainer": {
      borderColor: theme.palette.grey[300],
    },
  };

  if (!currentUser) {
    return null;
  }

  const tabConfigs = TabConfigs.filter(
    (tab) => !tab.globalPermission || currentUser.hasGlobalPermission(tab.globalPermission)
  );

  return (
    <Box>
      <Tabs value={selectedTab} variant="scrollable" sx={tabStyles}>
        {tabConfigs.map((tab) => (
          <Tab
            icon={tab.icon}
            iconPosition="start"
            value={tab.id}
            label={tab.title}
            component={Link}
            to={tab.route}
            sx={{ minHeight: "50px" }}
          />
        ))}
      </Tabs>
    </Box>
  );
}

TabHeader.Tab = ReviewTab;
