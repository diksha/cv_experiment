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
import { User } from "features/auth";

export interface Incident {
  // pk should be deprecated in favor of id
  pk: number;
  id: string;
  uuid: string;
  timestamp?: string;
  title?: string;
  // TODO: remove snake case version
  video_url?: string;
  videoUrl: string;
  // TODO: remove snake case version
  video_thumbnail_url?: string;
  thumbnailUrl: string;
  priority?: string;
  status?: string;
  camera_uuid: string;
  camera?: CameraType;
  // TODO: remove snake case version
  annotations_url: string;
  annotationsUrl?: string;
  incidentType?: IncidentType;
  incident_group_start_time_ms?: number;
  sub_incidents?: Incident[];
  actorIds: number[];
  bookmarked: boolean;
  alerted: boolean;
  // TODO: list_user_assigned in favor of assignees
  list_users_assigned?: User[];
  assignees: User[];
  dockerImageTag?: string;
}

export interface CameraType {
  id: number;
  key: string;
  name: string;
}

export interface IncidentType {
  id: string;
  key: string;
  name: string;
  backgroundColor: string;
}

export interface ZoneType {
  id: number;
  key: string;
  name: string;
}

export interface Comment {
  id: string;
  createdAt: string;
  text: string;
  owner: User;
  incident: Incident;
}

export enum LoadingStates {
  Idle = "idle",
  Pending = "pending",
}

export interface Series {
  key: string;
  incidentTypeCounts: {
    [key: string]: number;
  };
  priorityCounts: {
    lowPriorityCount: number;
    mediumPriorityCount: number;
    highPriorityCount: number;
  };
  statusCounts: {
    openCount: number;
    resolvedCount: number;
  };
}

export interface ObjectMapping {
  [key: string]: any;
}

export interface FilterOptions {
  fetchOnChange?: boolean;
}

export interface VoxelScore {
  label: string;
  value: number;
}

export interface ColorConfig {
  fill: string;
  text: string;
}
