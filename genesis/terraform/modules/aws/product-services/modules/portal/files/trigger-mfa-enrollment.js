function triggerMfaEnrollment(user, context, callback) {
    // checks mfa enrollment
    if (user.user_metadata && user.user_metadata.has_mfa) {
        context.multifactor = {
            provider: 'any',
            allowRememberBrowser: true,
        };
    }

    return callback(null, user, context);
}
