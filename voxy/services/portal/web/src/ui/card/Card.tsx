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
import styles from "./Card.module.css";
import { Spinner } from "ui";

interface Props {
  className?: string;
  title?: string;
  children?: React.ReactNode;
  loading?: boolean;
  noPadding?: boolean;
}

function CardBody({ children }: BodyProps) {
  return <div className="px-4 py-5 sm:p-6">{children}</div>;
}
function Card({ title, children, loading, className, noPadding }: Props) {
  const classes = classNames(className, "relative bg-white md:rounded-xl overflow-hidden");

  return (
    <div className={classes}>
      <LoadingOverlay loading={loading} />
      {title ? <div className="px-4 py-5 sm:px-6 font-bold border-b">{title}</div> : null}
      {noPadding ? children : <CardBody>{children}</CardBody>}
    </div>
  );
}

interface BodyProps {
  children: React.ReactNode;
}

Card.Body = CardBody;

export { Card };

export const LoadingOverlay = ({ loading }: any) => {
  const classes = classNames(styles.loadingOverlay, {
    [styles.displayed]: loading,
  });
  return (
    <div className={classes}>
      <Spinner />
    </div>
  );
};
