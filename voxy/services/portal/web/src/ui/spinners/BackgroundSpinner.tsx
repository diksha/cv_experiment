/*
 * Copyright 2020-2021 Voxel Labs, Inc.
 * All rights reserved.
 *
 * This document may not be reproduced, republished, distributed, transmitted,
 * displayed, broadcast or otherwise exploited in any manner without the express
 * prior written permission of Voxel Labs, Inc. The receipt or possession of this
 * document does not convey any rights to reproduce, disclose, or distribute its
 * contents, or to manufacture, use, or sell anything that it may describe, in
 * whole or in part.
 */
import { Spinner } from "ui";
import React, { useState } from "react";
import { useCurrentUser } from "features/auth";
import { Typography, Button, Box, Paper } from "@mui/material";
import { useTimeout } from "usehooks-ts";
import { Logout } from "@mui/icons-material";

const HELP_MESSAGE_TIMEOUT_MS = 10 * 1000;

export function BackgroundSpinner() {
  const { logout } = useCurrentUser();
  const [showHelp, setShowHelp] = useState(false);

  useTimeout(() => {
    setShowHelp(true);
  }, HELP_MESSAGE_TIMEOUT_MS);

  const handleLogout = () => {
    logout();
  };

  return (
    <div className="fixed inset-0 grid h-full content-center justify-center bg-gray-100">
      <div className="opacity-40">
        <Spinner />
      </div>
      {showHelp ? (
        <Paper
          sx={{
            position: "fixed",
            bottom: "20px",
            right: "20px",
            display: "flex",
            flexDirection: "column",
            gap: 2,
            padding: 2,
            width: "300px",
          }}
        >
          <Typography variant="h4" fontWeight="bold">
            Having trouble?
          </Typography>
          <Box>
            Try refreshing the page or contact us at <strong>support@voxelai.com</strong>.
          </Box>
          <Button variant="outlined" color="primary" size="small" onClick={handleLogout} startIcon={<Logout />}>
            Log out
          </Button>
        </Paper>
      ) : null}
    </div>
  );
}
