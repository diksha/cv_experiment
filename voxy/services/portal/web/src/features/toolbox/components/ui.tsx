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
import { Slideover } from "ui";
import { CaretLeft, CaretRight } from "phosphor-react";
import classNames from "classnames";

export const DrawerHeader = (props: { title: string; icon?: React.ReactNode; onClose: () => void }) => (
  <div className="flex gap-2 items-center text-white">
    <button className={classNames("rounded-full p-1 hover:bg-brand-gray-400")} onClick={props.onClose}>
      <CaretLeft className="h-8 w-8" />
    </button>
    <div className="font-bold text-xl">{props.title}</div>
  </div>
);

export function Drawer(props: { name: string; icon?: React.ReactNode; children?: React.ReactNode }) {
  const [open, setOpen] = useState(false);

  return (
    <div
      className={classNames(
        "overflow-hidden",
        "first:rounded-t-md last:rounded-b-md",
        "border-b border-opacity-20 border-brand-gray-500 last:border-b-0"
      )}
    >
      <button
        type="button"
        className={classNames("flex gap-4 w-full p-3 text-left items-center", "bg-brand-gray-400 hover:bg-opacity-80")}
        onClick={() => setOpen(!open)}
      >
        {props.icon && <div>{props.icon}</div>}
        <div className="flex-grow">{props.name}</div>
        <div className="flex-grow-0">
          <CaretRight className="h-6 w-6" />
        </div>
      </button>
      <Slideover
        open={open}
        onClose={() => setOpen(false)}
        title={<DrawerHeader title={props.name} icon={props.icon} onClose={() => setOpen(false)} />}
        hideCloseButton
        dark
      >
        <div className="text-white">{props.children}</div>
      </Slideover>
    </div>
  );
}
