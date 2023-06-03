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
import React, { useRef, useEffect } from "react";
import classNames from "classnames";

interface Props {
  children: any;
  onStuck: () => void;
  onUnstuck: () => void;
  sentinelClassName?: string;
  top?: number;
  zIndex?: number;
}

export const StickyWrapper = (props: Props) => {
  const ref = useRef(null);
  const { children, onStuck, onUnstuck, sentinelClassName } = props;
  const topOffset = props.top || 0;
  const zIndex: number | "auto" = props.zIndex || "auto";

  useEffect(() => {
    const cachedRef = ref.current;
    if (cachedRef) {
      const observer = new IntersectionObserver(
        ([e]) => {
          const elTop = e.target.getBoundingClientRect().top;
          const bottom25Percent = window.innerHeight * 0.75;
          const intersectingNearBottom = elTop >= bottom25Percent;
          if (!intersectingNearBottom) {
            if (e.intersectionRatio < 1) {
              onStuck();
            } else {
              onUnstuck();
            }
          }
        },
        {
          threshold: [1],
          rootMargin: `${topOffset * -1}px 0px 0px 0px`,
        }
      );

      observer.observe(cachedRef);

      // unmount
      return function () {
        observer.unobserve(cachedRef);
      };
    }
  }, [onStuck, onUnstuck, topOffset]);

  return (
    <>
      <div ref={ref} className={classNames("h-px max-h-px -mt-px mb-0 mx-0", sentinelClassName)}></div>
      <div style={{ top: topOffset, zIndex }} className="sticky">
        {children}
      </div>
    </>
  );
};
