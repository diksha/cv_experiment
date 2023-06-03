import { TabConfig } from "./types";
import { PAGE_REVIEW_QUEUE, PAGE_REVIEW_USERS, PAGE_REVIEW_HISTORY, EXPERIMENTAL_INCIDENTS_READ } from "features/auth";
import { ThumbUp, Science, People, History } from "@mui/icons-material";
import { ReviewTab } from "./enums";

export const TabConfigs: TabConfig[] = [
  {
    id: ReviewTab.Review,
    title: "Review",
    route: "/review",
    globalPermission: PAGE_REVIEW_QUEUE,
    active: false,
    icon: <ThumbUp />,
  },
  {
    id: ReviewTab.History,
    title: "History",
    globalPermission: PAGE_REVIEW_HISTORY,
    route: "/review/history",
    active: false,
    icon: <History />,
  },
  {
    id: ReviewTab.Experiments,
    title: "Experiments",
    globalPermission: EXPERIMENTAL_INCIDENTS_READ,
    route: "/review/experiments",
    active: false,
    icon: <Science />,
  },
  {
    id: ReviewTab.Users,
    title: "Users",
    globalPermission: PAGE_REVIEW_USERS,
    route: "/review/users",
    active: false,
    icon: <People />,
  },
];
