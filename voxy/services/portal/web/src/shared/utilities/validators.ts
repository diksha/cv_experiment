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
export const errors = {
  EMAIL_EMPTY: "EMAIL_EMPTY",
  EMAIL_LENGTH: "EMAIL_LENGTH",
  PASSWORD_EMPTY: "PASSWORD_EMPTY",
  PASSWORD_LENGTH: "PASSWORD_LENGTH",
  PASSWORD_CONFIRM_MISMATCH: "PASSWORD_CONFIRM_MISMATCH",
};

export const validateEmail = (email: string) => {
  if (email.length === 0) {
    return errors.EMAIL_EMPTY;
  } else if (email.length < 6) {
    return errors.EMAIL_LENGTH;
  }
  return "";
};

export const validatePassword = (password: string) => {
  if (password.length === 0) {
    return errors.PASSWORD_EMPTY;
  } else if (password.length < 8) {
    return errors.PASSWORD_LENGTH;
  }
  return "";
};

export const validatePasswordConfirmation = (password: string, confirmation: string) => {
  if (password !== confirmation) {
    return errors.PASSWORD_CONFIRM_MISMATCH;
  }
  return "";
};

export const authErrorMessage = (errorConstant: string): string => {
  switch (errorConstant) {
    case errors.EMAIL_EMPTY:
      return "Who are you?";
    case errors.EMAIL_LENGTH:
      return "That doesn't look like an email address";
    case errors.PASSWORD_EMPTY:
      return "You need this";
    case errors.PASSWORD_LENGTH:
      return "That's not long enough (min 8 characters)";
    case errors.PASSWORD_CONFIRM_MISMATCH:
      return "These passwords don't match";
    default:
      return "";
  }
};
