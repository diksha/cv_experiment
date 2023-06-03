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
import { Card } from "ui";
import { RiBookmarkFill } from "react-icons/ri";

export function SkeletonFeedItem() {
  return (
    <Card noPadding>
      <div className="p-4 border-b">
        <div className="rounded-md bg-brand-gray-100 w-24 h-3 animate-pulse"></div>
      </div>
      {[1, 2, 3, 4].map((key) => {
        return (
          <div key={key} className="flex pt-11 pb-4 px-4 md:pt-4 cursor-pointer relative border-b animate-pulse">
            <div className="flex gap-4">
              <div className="pr-4">
                <div className="w-44 h-24 rounded-md bg-brand-gray-100"></div>
              </div>
            </div>
            <div className="flex flex-col w-full">
              <div className="absolute top-2 left-0 pl-4 pr-3 md:px-0 md:static flex w-full justify-between flex-grow">
                <div className="rounded-md bg-brand-gray-100 w-48 h-6"></div>
                <div className="flex align-middle">
                  <div className="rounded-md bg-brand-gray-100 w-8 h-4 mr-1"></div>
                  <RiBookmarkFill className="text-brand-gray-100 h-4 w-4" />
                </div>
              </div>
              <div className="flex w-full justify-between">
                <div className="content-center">
                  <div className="rounded-md bg-brand-gray-100 w-24 h-3"></div>
                  <div className="rounded-md bg-brand-gray-100 w-12 h-3 mt-2"></div>
                </div>
              </div>
            </div>
          </div>
        );
      })}
      <div className="px-4 py-4 border-t animate-pulse">
        <div className="rounded-md bg-brand-gray-100 w-24 h-3 mx-auto"></div>
      </div>
    </Card>
  );
}
