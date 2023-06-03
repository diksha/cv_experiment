/*
 * Copyright 2022 Voxel Labs, Inc.
 * All rights reserved.
 *
 * This document may not be reproduced, republished, distributed, transmitted,
 * displayed, broadcast or otherwise exploited in any manner without the express
 * prior written permission of Voxel Labs, Inc. The receipt or possession of this
 * document does not convey any rights to reproduce, disclose, or distribute its
 * contents, or to manufacture, use, or sell anything that it may describe, in
 * whole or in part.
 */
import { isFilterValueEmpty, FilterBag } from "shared/filter";

/**
 * Generates a link to the /incidents page with the provided filters as query params.
 */
export const filteredIncidentsLink = (filterBag: FilterBag): string => {
  const searchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(filterBag)) {
    // Exclude filters with "empty" values
    if (!value || isFilterValueEmpty(value)) {
      continue;
    }

    // Value looks valid, attempt to serialize it
    // TODO: handle runtime serialization errors, for now let's not fail silently
    const valueJson = JSON.stringify(value.value);
    searchParams.append(key, valueJson);
  }
  return `/incidents?${searchParams.toString()}`;
};

interface SessionRecord {
  updatedAt: number;
}
export class SeenStateManager {
  static MAX_COUNT = 300;
  static HOUSEKEEPING_COUNT = 200;
  static KEY_PREFIX = "seen:incident:";

  static incidentSeenKey(incidentId: string) {
    return `${SeenStateManager.KEY_PREFIX}${incidentId}`;
  }

  static markAsSeen(incidentId: string) {
    const key = this.incidentSeenKey(incidentId);
    const value: SessionRecord = { updatedAt: Date.now() };
    localStorage.setItem(key, JSON.stringify(value));
    // Queue housekeeping asynchronously
    setTimeout(this.housekeeping, 0);
  }

  static isSeen(incidentId?: string): boolean {
    if (incidentId) {
      const key = this.incidentSeenKey(incidentId);
      return !!localStorage.getItem(key);
    }
    return false;
  }

  static clear() {
    Object.keys(localStorage)
      .filter((key) => key.startsWith(SeenStateManager.KEY_PREFIX))
      .forEach((key) => localStorage.removeItem(key));
  }

  /**
   * Deletes oldest N localStorage items until the total count <= HOUSEKEEPING_COUNT.
   *
   * This is a fairly expensive operation since we're parsing a lot of JSON, so we only
   * run the expensive stuff when the number of localStorage items exceeds MAX_COUNT, and
   * we only delete items until we hit HOUSEKEEPING_COUNT so that we don't erase all of
   * the users' seen statuses.
   *
   * The idea is we don't need to track this information forever, it is mostly a
   * convenience for users to track what they've seen within the last few hours or days,
   * so it's ok to lose this status for older items.
   *
   */
  static housekeeping() {
    const keys = Object.keys(localStorage).filter((key) => key.startsWith(SeenStateManager.KEY_PREFIX));
    const housekeepingNeeded = keys.length > SeenStateManager.MAX_COUNT;

    if (!housekeepingNeeded) {
      return;
    }

    keys
      .map((key) => {
        try {
          const item = JSON.parse(localStorage.getItem(key)!) as SessionRecord;
          return { key, sortValue: item.updatedAt };
        } catch {
          // Invalid item value, sortValue of 0 should result in this item being deleted
          return { key, sortValue: 0 };
        }
      })
      .sort((a, b) => a.sortValue - b.sortValue)
      .slice(SeenStateManager.HOUSEKEEPING_COUNT, keys.length - 1)
      .forEach((item) => {
        localStorage.removeItem(item.key);
      });
  }
}
