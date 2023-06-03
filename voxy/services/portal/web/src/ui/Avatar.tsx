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
import React, { useState } from "react";
import { IoIosPerson } from "react-icons/io";
import styles from "./Avatar.module.css";
import classNames from "classnames";

interface AvatarProps {
  url?: string;
  name?: string;
}

export function Avatar({ url, name }: AvatarProps) {
  const [valid, setValid] = useState(true);
  const handleError = () => {
    setValid(false);
  };

  const nameProvided = name && name.length > 0;
  const placeholder = nameProvided ? name![0] : <IoIosPerson />;
  const showPlaceholder = !url || !valid;

  return (
    <div className={styles.avatarWrapper}>
      {showPlaceholder && (
        <div
          className={classNames(
            "flex items-center justify-center",
            "h-10 w-10 rounded-full",
            "bg-brand-gray-400 ring-8 ring-white"
          )}
        >
          {placeholder}
        </div>
      )}
      {!showPlaceholder && (
        <img
          className={classNames(
            "flex items-center justify-center",
            "h-10 w-10 rounded-full",
            "bg-gray-400 ring-8 ring-white"
          )}
          src={url}
          alt={name}
          onError={handleError}
        />
      )}
    </div>
  );
}
