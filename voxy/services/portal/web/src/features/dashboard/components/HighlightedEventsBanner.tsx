import { Card } from "ui";
import { Link } from "react-router-dom";
import { Button } from "@mui/material";

interface HighlightedEventsBannerProps {
  message: string;
  link: string;
}

export function HighlightedEventsBanner({ message, link }: HighlightedEventsBannerProps) {
  return (
    <Card noPadding>
      <div className="flex flex-col sm:flex-row gap-4 p-3">
        <div className="flex flex-1 items-center font-bold">{message}</div>
        <div>
          <Button variant="contained" component={Link} to={link} data-ui-key="view-highlighted-events-banner-button">
            View Highlighted Events
          </Button>
        </div>
      </div>
    </Card>
  );
}
