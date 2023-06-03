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
import { Lock } from "phosphor-react";

export function TextInput(props: {
  name: string;
  label?: string;
  type?: string;
  placeholder?: string;
  value?: string;
  disabled?: boolean;
  error?: boolean;
  errors?: (string | null)[];
  className?: string;
  readOnly?: boolean;
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  // Filter out null and empty error strings
  const errors = (props.errors || []).filter((s: string | null): s is string => (s || "").length > 0);
  const errorState = props.error || errors.length > 0;
  return (
    <div>
      {props.label ? (
        <label htmlFor={props.name} className="block text-sm font-medium text-gray-700">
          {props.label}
        </label>
      ) : null}
      <div className="relative rounded-md shadow-sm">
        <input
          readOnly={props.readOnly}
          type={props.type || "text"}
          name={props.name}
          id={props.name}
          value={props.value}
          className={classNames(
            "block w-full pr-10 sm:text-sm rounded-md",
            errorState ? "border-brand-red-500" : "border-brand-gray-100",
            "focus:ring-blue-400 focus:border-blue-400",
            props.disabled && "text-gray-600 bg-gray-100 cursor-not-allowed",
            props.className ? props.className : null
          )}
          placeholder={props.placeholder}
          disabled={props.disabled}
          onChange={props.onChange}
        />
        {props.disabled ? (
          <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
            <Lock size={18} className="text-gray-400" aria-hidden="true" />
          </div>
        ) : null}
      </div>
      {errors.length > 0 ? (
        <div className="text-xs text-brand-red-500 pt-1 spacing-y-1">
          {errors.map((message: string, index: number) => (
            <div key={message}>{message}</div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
