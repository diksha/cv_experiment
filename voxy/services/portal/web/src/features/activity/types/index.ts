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

export enum ActivityType {
  Comment = "COMMENT",
  System = "SYSTEM",
  Assign = "ASSIGN",
}

export interface Activity {
  id: string;
  timestamp: string;
  type: ActivityType;
  data: SystemActivityData | AssignActivityData | CommentActivityData;
}

type CommentActivityData = {
  text: string;
  owner: User;
  incidentUuid?: string;
};

export class CommentActivity implements Activity {
  type = ActivityType.Comment;
  id: string;
  timestamp: string;
  data: CommentActivityData;

  constructor(id: string, timestamp: string, data: CommentActivityData) {
    this.id = id;
    this.timestamp = timestamp;
    this.data = data;
  }
}

type AssignActivityData = {
  text: string;
  owner: User;
  note: string;
  incidentUuid?: string;
};

export class AssignActivity implements Activity {
  type = ActivityType.Assign;
  id: string;
  timestamp: string;
  data: AssignActivityData;

  constructor(id: string, timestamp: string, data: AssignActivityData) {
    this.id = id;
    this.timestamp = timestamp;
    this.data = data;
  }
}

type SystemActivityData = {
  message: string;
  incidentUuid?: string;
};

export class SystemActivity implements Activity {
  type = ActivityType.System;
  id: string;
  timestamp: string;
  data: SystemActivityData;

  constructor(id: string, timestamp: string, data: SystemActivityData) {
    this.id = id;
    this.timestamp = timestamp;
    this.data = data;
  }
}
