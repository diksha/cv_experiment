const webpack = require("webpack");

const { GitRevisionPlugin } = require("git-revision-webpack-plugin");
const gitRevisionPlugin = new GitRevisionPlugin();

// config-overrides.js
module.exports = function override(config, env) {
  // New config, e.g. config.plugins.push...
  config.plugins.push(
    new webpack.DefinePlugin({
      COMMIT_HASH: JSON.stringify(gitRevisionPlugin.commithash()),
    })
  );

  // Required for @react-pdf/renderer to play nice with Webpack 5
  config.resolve.fallback = {
    ...config.resolve.fallback,
    "process/browser": require.resolve("process/browser"),
    zlib: require.resolve("browserify-zlib"),
    stream: require.resolve("stream-browserify"),
    util: require.resolve("util"),
    buffer: require.resolve("buffer"),
    asset: require.resolve("assert"),
    events: require.resolve("events"),
  };

  return config;
};
