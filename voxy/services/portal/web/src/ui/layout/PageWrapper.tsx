import { Breakpoint, Container } from "@mui/material";

/**
 * Used for constistent top/right/bottom/left padding for page-level content.
 */
interface PageWrapperProps {
  children: React.ReactNode;
  maxWidth?: Breakpoint;
  padding?: string | number;
  margin?: string | number;
}
export function PageWrapper({ children, maxWidth = "xl", padding = 2, margin = 0 }: PageWrapperProps) {
  return (
    <Container maxWidth={maxWidth} disableGutters sx={{ padding, margin }}>
      {children}
    </Container>
  );
}
