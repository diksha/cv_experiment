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
import React, { useState, useCallback, ReactNode } from "react";
import classNames from "classnames";
import { StickyWrapper } from "ui";

export function HeaderContainer(props: { children?: React.ReactNode; className?: string }) {
  const classes = classNames(
    "flex gap-4 items-center py-3 px-4 md:px-8 bg-white border-b border-brand-gray-050",
    props.className
  );

  return <div className={classes}>{props.children}</div>;
}

export function Header(props: { title: string | ReactNode; icon?: ReactNode; children?: ReactNode }) {
  return (
    <HeaderContainer>
      {props.icon ? <div className="hidden md:block">{props.icon}</div> : null}
      <div className="font-bold text-xl font-epilogue text-brand-gray-500 pt-1 flex-grow">{props.title}</div>
      {props.children}
    </HeaderContainer>
  );
}

interface StickyHeaderProps {
  className?: string;
  sentinelClassName?: string;
  children?: React.ReactNode;
  top?: number;
  zIndex?: number;
  flush?: boolean;
  shadow?: boolean;
  onStuck?: () => void;
  onUnStuck?: () => void;
}

export function StickyHeader(props: StickyHeaderProps) {
  const [headerSticky, setHeaderSticky] = useState(false);
  const classes = classNames(props.className, {
    "shadow-md": props.shadow && headerSticky,
  });

  const handleOnStuck = useCallback(() => {
    setHeaderSticky(true);
    props.onStuck?.();
  }, [props, setHeaderSticky]);

  const handleOnUnstuck = useCallback(() => {
    setHeaderSticky(false);
    props.onUnStuck?.();
  }, [props, setHeaderSticky]);

  return (
    <StickyWrapper
      top={props.top}
      zIndex={props.zIndex}
      onStuck={handleOnStuck}
      onUnstuck={handleOnUnstuck}
      sentinelClassName={props.sentinelClassName}
    >
      <div className={classes}>{props.children}</div>
    </StickyWrapper>
  );
}
