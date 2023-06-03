import React from "react";
import { useCurrentUser } from "features/auth";
import { SignOut } from "phosphor-react";
import { Logo, Card } from "ui";
import { LoadingButton } from "@mui/lab";
import { Link } from "react-router-dom";

export function AccountErrorPage() {
  const { logout, isLoading } = useCurrentUser();
  // TODO: add diagnostic info to the error message which users can share with us

  const handleLogout = () => {
    logout();
  };

  return (
    <div className="grid place-items-center h-screen p-4 bg-gray-100">
      <div>
        <Link to="/">
          <div className="block w-32 mx-auto pb-8">
            <Logo />
          </div>
        </Link>
        <Card>
          <div className="pb-8">Uh oh, your account appears to be misconfigured.</div>
          <ol className="list-decimal pl-8">
            <li>
              <p>Try logging out and signing in again: </p>
              <div className="flex justify-center py-4">
                <LoadingButton variant="outlined" onClick={handleLogout} loading={isLoading} startIcon={<SignOut />}>
                  Log out
                </LoadingButton>
              </div>
            </li>
            <li>If that doesn't work, please contact Voxel</li>
          </ol>
        </Card>
      </div>
    </div>
  );
}
