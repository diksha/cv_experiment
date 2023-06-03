import { useMutation } from "@apollo/client";
import { ArrowOutward } from "@mui/icons-material";
import { Button, useTheme } from "@mui/material";
import { CurrentUserSiteUpdate, CurrentUserSiteUpdateVariables } from "__generated__/CurrentUserSiteUpdate";
import { OrgSite } from "features/executive-dashboard";
import { CURRENT_USER_SITE_UPDATE } from "features/organizations";
import { useNavigate } from "react-router-dom";

interface GoToSiteDashboardButtonProps {
  site: OrgSite;
}
export function GoToSiteDashboardButton({ site }: GoToSiteDashboardButtonProps) {
  const theme = useTheme();
  const navigate = useNavigate();

  const [userUpdate] = useMutation<CurrentUserSiteUpdate, CurrentUserSiteUpdateVariables>(CURRENT_USER_SITE_UPDATE);

  const handleClick = async () => {
    try {
      await userUpdate({
        variables: {
          siteId: site.id,
        },
      });
      navigate("/dashboard", { replace: true });
      navigate(0);
    } catch {
      alert("Sorry, something went wrong while switching sites");
    }
  };

  return (
    <Button
      variant="outlined"
      endIcon={<ArrowOutward sx={{ height: 16, width: 16 }} />}
      sx={{
        border: `1px solid ${theme.palette.grey[300]}`,
        fontWeight: "400",
        height: "34px",
        borderRadius: "6px",
        lineHeight: "1.2",
      }}
      onClick={handleClick}
    >
      Go to Site Dashboard
    </Button>
  );
}
