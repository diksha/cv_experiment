import classNames from "classnames";

import React, { useState } from "react";
import { Spinner } from "ui";
import { IconButton, Box, Button, Dialog } from "@mui/material";
import { CloseOutlined, SmsOutlined, FileCopyOutlined, MailOutlined } from "@mui/icons-material";
import { useMutation } from "@apollo/client";
import { CREATE_SHARE_LINK } from "features/incidents";
import { IncidentCreateShareLink, IncidentCreateShareLinkVariables } from "__generated__/IncidentCreateShareLink";

import { Share } from "phosphor-react";

interface EventShareLinkButtonProps {
  incidentId: string;
  incidentTitle: string;
  fullWidth?: boolean;
}

export function EventShareLinkButton(props: EventShareLinkButtonProps) {
  const { incidentId, incidentTitle, fullWidth } = props;

  const [open, setOpen] = useState(false);
  const [shareLink, setShareLink] = useState("");
  const shareData = { url: shareLink };
  const canShare = window.navigator.canShare?.(shareData);
  const [createShareLink, { loading }] = useMutation<IncidentCreateShareLink, IncidentCreateShareLinkVariables>(
    CREATE_SHARE_LINK,
    {
      onCompleted: (data) => {
        setShareLink(data?.incidentCreateShareLink?.shareLink || "");
      },
    }
  );

  const [copyButtonText, setCopyButtonText] = useState("Copy Link");

  const onOpen = () => {
    setOpen(true);
    createShareLink({
      variables: {
        incidentId: incidentId,
      },
    });
  };

  const handleCopyClick = async () => {
    await window.navigator.clipboard?.writeText?.(shareLink);
    setCopyButtonText("Copied!");

    setTimeout(() => {
      setCopyButtonText("Copy Link");
    }, 2000);
  };

  const handleEmailClick = async () => {
    const href = `mailto:?body=${shareLink}`;
    window.location.assign(href);
  };

  const handleShareClick = async () => {
    if (canShare) {
      await window.navigator.share?.(shareData);
    }
  };

  return (
    <>
      <Button variant="contained" onClick={onOpen} startIcon={<Share />} fullWidth={fullWidth}>
        Share Event Link
      </Button>
      <Dialog
        open={open}
        onClose={() => {
          setOpen(false);
        }}
      >
        <Box sx={{ padding: "1rem", maxWidth: "400px" }}>
          <div className="mx-2 mb-2">
            <div className="flex w-full items-center justify-between">
              <div className="text-xl font-bold text-brand-gray-500 font-epilogue">Share {incidentTitle}</div>
              <IconButton
                sx={{ marginTop: "-8px", marginRight: "-8px" }}
                onClick={() => {
                  setOpen(false);
                }}
              >
                <CloseOutlined />
              </IconButton>
            </div>
            <div className="my-2">
              <div>
                Copy and share link by email or text message. Note: this link will expire after 3 days for security
                reasons.
              </div>
            </div>
            <div className="w-full">
              {loading ? (
                <div className="flex justify-center items-center w-full">
                  <Spinner />
                </div>
              ) : (
                <div className="flex flex-col gap-3">
                  <div className="w-full flex rounded-md border-2">
                    <input
                      readOnly={true}
                      type="text"
                      name="event"
                      id="event"
                      value={shareLink}
                      className={classNames(
                        "w-full sm:text-sm rounded-md",
                        "focus:ring-blue-400 focus:border-blue-400",
                        "py-3 border-none"
                      )}
                    />
                  </div>
                  <Box
                    sx={{
                      display: "flex",
                      gap: ".75rem",
                      flexDirection: {
                        xs: "column",
                        sm: "row",
                      },
                    }}
                  >
                    <Button
                      variant="contained"
                      fullWidth
                      onClick={handleCopyClick}
                      startIcon={<FileCopyOutlined />}
                      data-ui-key="share-link-copy-btn"
                    >
                      {copyButtonText}
                    </Button>
                    {canShare ? (
                      <Button
                        variant="contained"
                        fullWidth
                        onClick={handleShareClick}
                        startIcon={<SmsOutlined />}
                        data-ui-key="share-link-share-btn"
                      >
                        Email Or SMS
                      </Button>
                    ) : (
                      <Button
                        variant="contained"
                        fullWidth
                        onClick={handleEmailClick}
                        startIcon={<MailOutlined />}
                        data-ui-key="share-link-email-btn"
                      >
                        Email
                      </Button>
                    )}
                  </Box>
                </div>
              )}
            </div>
          </div>
        </Box>
      </Dialog>
    </>
  );
}
