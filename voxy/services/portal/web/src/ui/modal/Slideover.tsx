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
import React, { Fragment } from "react";
import classNames from "classnames";
import { Dialog, Transition } from "@headlessui/react";
import { X } from "phosphor-react";

function Title(props: { title: JSX.Element | string }) {
  return (
    <div className="px-4 sm:px-6">
      <Dialog.Title className="text-xl font-bold text-brand-gray-500 font-epilogue">{props.title}</Dialog.Title>
    </div>
  );
}

function Slideover(props: {
  title?: JSX.Element | string;
  open: boolean;
  onClose: () => void;
  children: React.ReactNode;
  hideCloseButton?: boolean;
  dark?: boolean;
  noPadding?: boolean;
}) {
  const wrapperClasses = classNames(
    "h-full flex flex-col shadow-xl overflow-y-scroll",
    props.dark ? "bg-brand-gray-500" : "bg-white",
    props.noPadding ? "" : "py-6"
  );
  const contentClasses = classNames("flex flex-col flex-1 relative", props.noPadding ? "" : "mt-6 px-4 sm:px-6");

  return (
    <Transition.Root show={props.open} as={Fragment}>
      <Dialog as="div" className="fixed inset-0 overflow-hidden z-[1300]" onClose={props.onClose}>
        <div className="absolute inset-0 overflow-hidden">
          <Transition.Child
            as={Fragment}
            enter="ease-in-out duration-300"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="ease-in-out duration-300"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <Dialog.Overlay className="absolute inset-0 bg-gray-500 bg-opacity-75 transition-opacity" />
          </Transition.Child>
          <div className="fixed inset-y-0 right-0 pl-10 max-w-full flex">
            <Transition.Child
              as={Fragment}
              enter="transform transition ease-in-out duration-300 sm:duration-500"
              enterFrom="translate-x-full"
              enterTo="translate-x-0"
              leave="transform transition ease-in-out duration-300 sm:duration-500"
              leaveFrom="translate-x-0"
              leaveTo="translate-x-full"
            >
              <div className="relative w-screen max-w-md">
                <Transition.Child
                  as={Fragment}
                  enter="ease-in-out duration-300"
                  enterFrom="opacity-0"
                  enterTo="opacity-100"
                  leave="ease-in-out duration-300"
                  leaveFrom="opacity-100"
                  leaveTo="opacity-0"
                >
                  <div className="absolute top-0 left-0 -ml-8 pt-4 pr-2 flex sm:-ml-10 sm:pr-4">
                    {props.hideCloseButton ? null : (
                      <button
                        type="button"
                        className="rounded-md text-gray-300 hover:text-white focus:outline-none focus:ring-2 focus:ring-white"
                        onClick={props.onClose}
                      >
                        <span className="sr-only">Close panel</span>
                        <X size={32} aria-hidden="true" />
                      </button>
                    )}
                  </div>
                </Transition.Child>
                <div className={wrapperClasses}>
                  {props.title ? <Title title={props.title} /> : null}
                  <div className={contentClasses}>{props.children}</div>
                </div>
              </div>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition.Root>
  );
}

Slideover.Footer = (props: { children: React.ReactNode; noPadding?: boolean }) => {
  const wrapperClasses = classNames(props.noPadding ? "" : "px-4 py-4");
  return <div className={wrapperClasses}>{props.children}</div>;
};

export { Slideover };
