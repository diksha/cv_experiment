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

import {
  GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_EmptyRangeFeedItem,
  GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem,
  GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node,
  GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets,
} from "__generated__/GetCurrentZoneIncidentFeed";

export type DailyIncidentsFeedItemData =
  GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem;
export type DailyIncidentsFeedItemTimeBucketData =
  GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_DailyIncidentsFeedItem_timeBuckets;
export type EmptyRangeFeedItemData =
  GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node_EmptyRangeFeedItem;
export type FeedItemData = GetCurrentZoneIncidentFeed_currentUser_zone_incidentFeed_edges_node;
