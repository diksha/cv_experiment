function addUserMetadataToAuth0UserAPI(user, context, callback) {
    // adds to metadata returned
    if (user.user_metadata && user.user_metadata.has_mfa) {
        context.idToken.has_mfa = user.user_metadata.has_mfa;
    }
    context.idToken.connection = context.connection;
    
    return callback(null, user, context);
}
