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
import React from "react";
import classNames from "classnames";
import styles from "./Spinner.module.css";

export function Spinner(props: { className?: string; white?: boolean; small?: boolean }) {
  const wrapperClasses = classNames(styles.wrapper, props.className, props.small ? "w-5 h-5" : null);
  const spinnerClasses = classNames(styles.spinner, props.white ? styles.white : null, props.small ? "w-5 h-5" : null);
  return (
    <div className={wrapperClasses}>
      <div className={spinnerClasses}></div>
    </div>
  );
}
