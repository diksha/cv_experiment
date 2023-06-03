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
import { ReactNode, useRef, useEffect, useState, useCallback } from "react";
import classNames from "classnames";
import { CaretLeft, CaretRight } from "phosphor-react";

interface StatisticListProps {
  children: ReactNode;
  scrollContainerClasses?: string;
}

export function StatisticList(props: StatisticListProps) {
  const [hasScrollbar, setHasScrollbar] = useState<boolean>();
  const [disableLeftArrow, setDisableLeftArrow] = useState<boolean>(true);
  const [disableRightArrow, setDisableRightArrow] = useState<boolean>(false);
  const listRef = useRef<HTMLDivElement>(null);

  const handleScrollCheck = () => {
    const listEl = listRef?.current;
    if (!listEl) {
      return;
    }
    const gap = listEl.scrollWidth - listEl.offsetWidth;
    setDisableLeftArrow(listEl.scrollLeft === 0);
    setDisableRightArrow(gap !== 0 && listEl.scrollLeft === gap);
  };

  const handleScrollbarCheck = () => {
    const listEl = listRef?.current;
    if (listEl) {
      const scrollable = listEl.scrollWidth > listEl.clientWidth;
      setHasScrollbar(scrollable);
    }
  };

  const handleScrollButtonClick = useCallback((isBackward) => {
    const listEl = listRef?.current;
    if (!listEl) {
      return;
    }
    listEl.scroll({
      left: listEl.scrollLeft + (isBackward ? -1 : 1) * Math.floor(listEl.scrollWidth / 4),
      behavior: "smooth",
    });
  }, []);

  useEffect(() => {
    handleScrollbarCheck();
    window.addEventListener("resize", handleScrollbarCheck);
    return () => {
      window.removeEventListener("resize", handleScrollbarCheck);
    };
  }, []);

  return (
    <>
      <div
        ref={listRef}
        onScroll={handleScrollCheck}
        className={classNames(props.scrollContainerClasses, "overflow-auto whitespace-nowrap")}
      >
        {props.children}
      </div>
      <div className={classNames(hasScrollbar ? "flex" : "hidden", "mt-3 justify-center items-center gap-1")}>
        <button
          disabled={disableLeftArrow}
          onClick={() => handleScrollButtonClick(true)}
          className={classNames(
            "rounded-full border-2 text-center w-5 h-5",
            disableLeftArrow
              ? "border-brand-gray-100 text-brand-gray-200 cursor-not-allowed"
              : "border-brand-gray-500 bg-brand-gray-100"
          )}
        >
          <CaretLeft className="mx-auto" size={11} />
        </button>
        <button
          disabled={disableRightArrow}
          onClick={() => handleScrollButtonClick(false)}
          className={classNames(
            "rounded-full border-2 text-center w-5 h-5",
            disableRightArrow
              ? "border-brand-gray-100 text-brand-gray-200 cursor-not-allowed"
              : "border-brand-gray-500 bg-brand-gray-100"
          )}
        >
          <CaretRight className="mx-auto" size={11} />
        </button>
      </div>
    </>
  );
}
