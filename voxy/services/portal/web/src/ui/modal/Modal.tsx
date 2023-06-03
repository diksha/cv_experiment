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
import classNames from "classnames";
import React from "react";
import styles from "./Modal.module.css";
import { Link } from "react-router-dom";
import { Dialog } from "@mui/material";

export const ModalBody = (props: any) => {
  const classes = classNames(styles.modalBody, {
    [styles.active]: props.active,
  });

  return <div className={classes}>{props.children}</div>;
};

export const ModalFooter = (props: any) => {
  const { active, className, rightAlign, ...extras } = props;
  const classes = classNames(styles.modalFooter, className, {
    [styles.active]: active,
    [styles.rightAlign]: rightAlign,
  });

  return (
    <div className={classes} {...extras}>
      {props.children}
    </div>
  );
};

type ModalProps = {
  open: boolean;
  fitContent?: boolean;
  onClose: (open: false) => void;
  children: React.ReactNode;
  overrides?: string;
  showSupportFooter?: boolean;
};

export function Modal({ open, onClose, children, fitContent, overrides, showSupportFooter }: ModalProps) {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      scroll="body"
      classes={{ paper: classNames("p-0", fitContent ? "" : "sm:max-w-sm sm:w-full", overrides) }}
    >
      {children}
      {showSupportFooter ? <SupportFooter /> : null}
    </Dialog>
  );
}

function SupportFooter() {
  return (
    <div className="p-6 bg-brand-yellow-100 text-brand-yellow-700 rounded-bl-lg rounded-br-lg">
      Having trouble? Visit our{" "}
      <Link to="/support" className="font-bold underline">
        support page
      </Link>{" "}
      or contact us at{" "}
      <a target="_blank" rel="noopener noreferrer" href="mailto:support@voxelai.com" className="font-bold underline">
        support@voxelai.com
      </a>
      .
    </div>
  );
}
