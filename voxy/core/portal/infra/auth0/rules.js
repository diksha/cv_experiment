/* eslint-disable */
async function emailDomainWhitelist(user, context, callback) {
  const axios = require("axios@0.19.2");

  const whitelist = ["voxelai.com"]; //authorized domains
  const userHasAccess = whitelist.some(function (domain) {
    if (context.connection === "Username-Password-Authentication") {
      return true;
    }

    const emailSplit = user.email.split("@");
    return emailSplit[emailSplit.length - 1].toLowerCase() === domain;
  });

  if (userHasAccess) {
    return callback(null, user, context);
  }

  const options = {
    method: "POST",
    url: `https://${auth0.domain}/oauth/token`,
    headers: {
      "content-type": "application/json",
    },
    data: {
      client_id: configuration.DELETE_USERS_CLIENT_ID,
      client_secret: configuration.DELETE_USERS_CLIENT_SECRET,
      audience: `https://${auth0.domain}/api/v2/`,
      grant_type: "client_credentials",
    },
  };

  try {
    const tokenResponse = await axios(options);
    const accessToken = tokenResponse.data.access_token;
    const userId = encodeURIComponent(user.user_id);
    const deleteUserOptions = {
      method: "DELETE",
      url: `https://${auth0.domain}/api/v2/users/${userId}`,
      headers: { Authorization: `Bearer ${accessToken}` },
    };
    await axios(deleteUserOptions);
    return callback(new UnauthorizedError("Access denied."));
  } catch (err) {
    // handle error
    console.log(err);
    return callback(new UnauthorizedError("Access denied."));
  }
}
