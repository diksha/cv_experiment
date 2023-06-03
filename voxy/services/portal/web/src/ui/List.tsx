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
import { Transition } from "@headlessui/react";

interface ListItemProps {
  onClick: () => void;
  active: boolean;
  rowRender: React.ReactNode;
  activeRender?: React.ReactNode;
}
export function ListItem({ onClick, active, rowRender, activeRender }: ListItemProps) {
  return (
    <div className="flex border-t first:border-t-0 hover:bg-gray-50">
      <div className="flex-grow">
        <div className="flex gap-4 p-4 cursor-pointer items-center" onClick={onClick}>
          <div className="w-full">{rowRender}</div>
        </div>
        {activeRender && (
          <Transition
            show={active}
            enter="transition-all duration-100"
            enterFrom="opacity-0 h-0"
            enterTo="opacity-100 h-64"
            leave="transition-all duration-100"
            leaveFrom="opacity-100 h-64"
            leaveTo="opacity-0 h-0"
          >
            <div className="flex gap-4 p-4 pt-0 flex-col lg:flex-row h-auto shadow-xl">{activeRender}</div>
          </Transition>
        )}
      </div>
    </div>
  );
}

interface ListProps {
  children: React.ReactNode;
}
export function List({ children }: ListProps) {
  return <div>{children}</div>;
}

List.Item = ListItem;
