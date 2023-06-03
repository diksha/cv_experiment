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

const { createProxyMiddleware } = require("http-proxy-middleware");
const BACKEND_HOST = "http://localhost:9001";

module.exports = function (app) {
  app.use("/api", createProxyMiddleware({ target: BACKEND_HOST, changeOrigin: true }));
  app.use("/admin", createProxyMiddleware({ target: BACKEND_HOST, changeOrigin: true }));
  app.use("/static/admin", createProxyMiddleware({ target: BACKEND_HOST, changeOrigin: true }));
  app.use("/graphql", createProxyMiddleware({ target: BACKEND_HOST, changeOrigin: true }));
  app.use("/static/graphene_django", createProxyMiddleware({ target: BACKEND_HOST, changeOrigin: true }));
  app.use("/internal/backend", createProxyMiddleware({ target: BACKEND_HOST, changeOrigin: true }));
  app.use("/static/dash", createProxyMiddleware({ target: BACKEND_HOST, changeOrigin: true }));
};
