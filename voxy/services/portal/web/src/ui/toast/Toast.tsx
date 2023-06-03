import React, { ReactNode } from "react";
import { Transition } from "@headlessui/react";
import { WarningOctagon, CheckCircle } from "phosphor-react";

import { Toast } from "react-hot-toast";

const BaseToast = (props: { toast: Toast; children: ReactNode }) => (
  <Transition
    appear
    show={props.toast.visible}
    className="transform"
    enter="transition-all duration-150"
    enterFrom="opacity-0 scale-50"
    enterTo="opacity-100 scale-100"
    leave="transition-all duration-150"
    leaveFrom="opacity-100 scale-100"
    leaveTo="opacity-0 scale-75"
  >
    {props.children}
  </Transition>
);

export const SuccessToast = (props: { toast: Toast; children: ReactNode }) => (
  <BaseToast toast={props.toast}>
    <div className="bg-brand-green-100 rounded-full px-8 py-4 flex justify-center items-center shadow-md">
      <CheckCircle className="w-6 h-6 m-1 text-brand-green-900" weight="fill" />
      <div className="text-brand-green-900 font-normal mx-1">{props.children}</div>
    </div>
  </BaseToast>
);

export const ErrorToast = (props: { toast: Toast; children: ReactNode }) => (
  <BaseToast toast={props.toast}>
    <div className="bg-brand-red-100 rounded-full px-8 py-4 flex justify-center items-center shadow-md">
      <WarningOctagon className="w-6 h-6 m-1 text-red-green-900" weight="fill" />
      <div className="text-brand-red-900 font-normal mx-1">{props.children}</div>
    </div>
  </BaseToast>
);
