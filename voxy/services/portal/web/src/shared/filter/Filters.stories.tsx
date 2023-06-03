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
import { Story } from "@storybook/react";
import {
  Filters,
  FILTER_VALUE_HIGH_PRIORITY,
  FILTER_VALUE_UNASSIGNED_STATUS,
  FILTER_VALUE_OPEN_AND_ASSIGNED_STATUS,
  FILTER_VALUE_ASSIGNED_TO_ME,
} from ".";

const config = {
  title: "Filters",
  component: Filters,
};
export default config;

const Template: Story<React.ComponentProps<typeof Filters>> = (args) => <Filters {...args} />;

export const Default = Template.bind({});
export const HasFilters = Template.bind({});

Default.args = {};

HasFilters.args = {
  values: {
    assignee: {
      value: [FILTER_VALUE_ASSIGNED_TO_ME],
    },
    status: {
      value: [FILTER_VALUE_UNASSIGNED_STATUS, FILTER_VALUE_OPEN_AND_ASSIGNED_STATUS],
    },
    priority: {
      value: [FILTER_VALUE_HIGH_PRIORITY],
    },
  },
};
