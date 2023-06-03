import { useState, useEffect, ReactNode } from "react";
import { Link } from "react-router-dom";
import { Modal } from "ui";
import { Box, Button, Typography, useTheme } from "@mui/material";
import { Campaign } from "@mui/icons-material";

interface AnnouncementProps {
  demoUrl?: string;
  tryUrl?: string;
  tryLabel?: string;
  date?: string;
  id: string;
  title: string;
  children: ReactNode;
}

const ANNOUNCEMENT_KEY = "seen:announcement:";

export function Announcement(props: AnnouncementProps) {
  const theme = useTheme();
  const [isOpened, setIsOpened] = useState<boolean>(false);

  const handleClose = () => {
    localStorage.setItem(`${ANNOUNCEMENT_KEY}${props.id}`, "true");
    setIsOpened(false);
  };

  useEffect(() => {
    const seen = !!localStorage.getItem(`${ANNOUNCEMENT_KEY}${props.id}`);
    setIsOpened(!seen);
  }, [props.id]);

  return (
    <Modal open={isOpened} onClose={handleClose} fitContent>
      <Box
        sx={{
          backgroundColor: theme.palette.primary.main,
          height: "94px",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <Campaign sx={{ color: "#ffffff", width: "46px", height: "46px" }} />
      </Box>
      <Box sx={{ padding: "32px 36px 0" }}>
        {props.demoUrl && (
          <img
            className="rounded-3xl mb-6 border-4 border-brand-gray-200 w-full"
            alt=""
            style={{ maxWidth: "560px" }}
            src={props.demoUrl}
          />
        )}
        <Box sx={{ width: "100%", maxWidth: "560px" }}>
          <Typography fontWeight="bold" fontSize={20}>
            <span style={{ color: "#00B69B" }}>New!&nbsp;</span>
            {props.title}
          </Typography>
          {props.children}
        </Box>
      </Box>
      {props.tryUrl && (
        <Box sx={{ marginTop: "24px", marginBottom: "32px", padding: "0 36px", textAlign: "center" }}>
          <Button
            component={Link}
            to={props.tryUrl}
            onClick={handleClose}
            variant="contained"
            sx={{ backgroundColor: theme.palette.primary.main, padding: "12px 32px", borderRadius: "4px" }}
          >
            {props.tryLabel || "Try It Out"}
          </Button>
        </Box>
      )}
    </Modal>
  );
}

export function CurrentAnnouncements() {
  // To display an announcement to users, update this component to return an <Announcement /> component
  // TODO: fetch announcements from DB/API and dynamically render them here

  return null;

  // return (
  //   <Announcement
  //     id="announcement-new-dashboard"
  //     title="Voxel Dashboard"
  //     tryLabel="View New Dashboard"
  //     tryUrl="/dashboard"
  //   >
  //     <Box sx={{ margin: "16px 0 24px" }}>
  //       <Typography fontSize={16}>
  //         It's now easier than ever to identify problem areas, and measure performance with new Voxel Dashboard.
  //       </Typography>
  //     </Box>
  //     <Box>
  //       <List sx={{ listStyleType: "disc" }}>
  //         <ListItem sx={{ display: "list-item", marginLeft: "24px", paddingLeft: "4px", fontSize: "16px" }}>
  //           Easily measure progress towards injury-zero with Trends.
  //         </ListItem>
  //         <ListItem sx={{ display: "list-item", marginLeft: "24px", paddingLeft: "4px", fontSize: "16px" }}>
  //           Quickly access the most notable video clips with Highlight Incidents.
  //         </ListItem>
  //         <ListItem sx={{ display: "list-item", marginLeft: "24px", paddingLeft: "4px", fontSize: "16px" }}>
  //           Identify which days, times or locations have the most incidents in just a few clicks.
  //         </ListItem>
  //       </List>
  //     </Box>
  //   </Announcement>
  // );
}
