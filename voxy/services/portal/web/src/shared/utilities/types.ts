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
export function isNumber(value: any): boolean {
  return value != null && value !== "" && !isNaN(Number(value.toString()));
}

/**
 * Filters an array to remove null and undefined values.
 * @param values array of values to filter
 * @returns array of values of type T
 */
export function filterNullValues<T>(
  values: T[] | (T | null)[] | (T | undefined)[] | (T | null | undefined)[] | null | undefined
): T[] {
  const output: T[] = [];
  for (const value of values || []) {
    if (value !== null && value !== undefined) {
      output.push(value);
    }
  }
  return output;
}
