import React, { useCallback } from "react";
import { useCurrentUser } from "features/auth";
import { SignOut } from "phosphor-react";
import { LoadingButton } from "@mui/lab";

export function LogoutPage() {
  const { logout, isLoading } = useCurrentUser();

  const handleLogout = useCallback(() => {
    logout({ returnTo: window.location.origin });
  }, [logout]);

  return (
    <div className="flex justify-center items-center h-screen w-full">
      <LoadingButton
        variant="contained"
        onClick={handleLogout}
        data-ui-key="logout"
        loading={isLoading}
        startIcon={<SignOut />}
      >
        Log Out
      </LoadingButton>
    </div>
  );
}
