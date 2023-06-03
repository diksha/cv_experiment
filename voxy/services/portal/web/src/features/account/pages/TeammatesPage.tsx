import { Helmet } from "react-helmet-async";
import { useState, useCallback } from "react";
import { PageWrapper, TopNavBack } from "ui";
import { AccountTab, OrganizationUserListV2, TabbedContainer, UserInviteModal } from "features/account";
import { Box, Button, Card, Typography, useMediaQuery, useTheme } from "@mui/material";
import { useCurrentUser, USERS_INVITE } from "features/auth";
import { UserPlus } from "phosphor-react";

export function TeammatesPage() {
  const { currentUser } = useCurrentUser();
  const theme = useTheme();
  const [isUserInviteModalOpen, setIsUserInviteModalOpen] = useState(false);
  const mdBreakpoint = useMediaQuery(theme.breakpoints.up("md"));

  const handleOpen = useCallback(
    (open) => {
      return () => {
        return setIsUserInviteModalOpen(open);
      };
    },
    [setIsUserInviteModalOpen]
  );

  const inviteBtn = currentUser?.hasZonePermission(USERS_INVITE) ? (
    <Button color="primary" variant="outlined" onClick={handleOpen(true)} startIcon={<UserPlus />}>
      Invite Teammates
    </Button>
  ) : null;

  return (
    <>
      <Helmet>
        <title>Teammates - Voxel</title>
      </Helmet>
      <TopNavBack mobTitle="Teammates" mobTo="/account">
        {inviteBtn}
      </TopNavBack>
      <PageWrapper maxWidth="md" padding={mdBreakpoint ? 2 : 0} margin="0 auto">
        {mdBreakpoint ? (
          <TabbedContainer activeTab={AccountTab.Teammates}>
            <Box sx={{ display: "flex", justifyContent: "space-between", padding: "16px 0" }}>
              <Typography variant="h2">Teammates</Typography>
              {inviteBtn}
            </Box>
            <OrganizationUserListV2 />
          </TabbedContainer>
        ) : (
          <Card sx={{ padding: 2, width: "100%", borderRadius: 0 }}>
            <OrganizationUserListV2 />
          </Card>
        )}
      </PageWrapper>
      <UserInviteModal isUserInviteModalOpen={isUserInviteModalOpen} onClose={handleOpen(false)} />
    </>
  );
}
