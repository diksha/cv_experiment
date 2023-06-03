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
import { Faders } from "phosphor-react";
import { FilterButton } from ".";

const config = {
  title: "Filter/Button",
  component: FilterButton,
};
export default config;

const Template: Story<React.ComponentProps<typeof FilterButton>> = (args) => <FilterButton {...args} />;

export const Default = Template.bind({});
export const WithIconAndIndicator = Template.bind({});

Default.args = {
  label: "Assigned to me",
};

WithIconAndIndicator.args = {
  label: "All Filters",
  icon: <Faders size={16} className="mr-1 inline-block" />,
  indicator: 2,
};
