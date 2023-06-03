import { Helmet } from "react-helmet-async";
import { Box, Card, useTheme, useMediaQuery, Typography } from "@mui/material";
import { PageWrapper, TopNavBack } from "ui";
import { OpenInNewOutlined } from "@mui/icons-material";
import { AccountTab, TabbedContainer } from "features/account";

const SUPPORT_LINKS = [
  { title: "FAQs", href: "https://docs.google.com/document/d/1CotPPyLrMwmnY7KJSOfw9DNV8GADQ-weJ_7up1y46pA/edit" },
  { title: "Videos", href: "https://www.youtube.com/playlist?list=PLOkzpcQgZJyZQNpWmDeAUjbgoR98tIYzq" },
  { title: "Support Request", href: "https://forms.clickup.com/36001183/f/12ancz-22860/YWJPXICAK60GQ6DP9P" },
];

export function SupportPage() {
  const theme = useTheme();
  const mdBreakpoint = useMediaQuery(theme.breakpoints.up("md"));

  const content = (
    <>
      <Box component="ul" sx={{ listStyleType: "disc", paddingLeft: 2 }}>
        {SUPPORT_LINKS.map((link) => (
          <li key={link.title}>
            <Box
              component="a"
              href={link.href}
              target="_blank"
              rel="noreferrer"
              sx={{ display: "block", textDecoration: "underline", paddingBottom: 1 }}
            >
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                {link.title}
                <OpenInNewOutlined sx={{ height: "1rem", width: "1rem" }} />
              </Box>
            </Box>
          </li>
        ))}
      </Box>
    </>
  );

  return (
    <>
      <Helmet>
        <title>Support - Voxel</title>
      </Helmet>
      <TopNavBack mobTitle="Support" mobTo="/account" />
      <PageWrapper maxWidth="md" padding={mdBreakpoint ? 2 : 0} margin="0 auto">
        {mdBreakpoint ? (
          <TabbedContainer activeTab={AccountTab.Support}>
            <Typography variant="h2" paddingTop={2} paddingBottom={3}>
              Support
            </Typography>
            {content}
          </TabbedContainer>
        ) : (
          <Card sx={{ padding: 2, width: "100%", borderRadius: 0 }}>{content}</Card>
        )}
      </PageWrapper>
    </>
  );
}
