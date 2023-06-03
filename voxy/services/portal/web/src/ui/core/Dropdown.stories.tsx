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
import { Story } from "@storybook/react";
import { Dropdown } from ".";

const config = {
  title: "Core/Dropdown",
  component: Dropdown,
};
export default config;

const people = [
  { id: 1, name: "Wade Cooper" },
  { id: 2, name: "Arlene Mccoy" },
  { id: 3, name: "Devon Webb" },
  { id: 4, name: "Tom Cook" },
  { id: 5, name: "Tanya Fox" },
  { id: 6, name: "Hellen Schmidt" },
  { id: 7, name: "Caroline Schultz" },
  { id: 8, name: "Mason Heaney" },
  { id: 9, name: "Claudie Smitham" },
  { id: 10, name: "Emil Schaefer" },
];

const Template: Story<React.ComponentProps<typeof Dropdown>> = (args) => (
  <div className="w-60 p-4">
    <Dropdown {...args} />
  </div>
);

const AllTemplate: Story<React.ComponentProps<typeof Dropdown>> = (args) => (
  <div>
    <div className="mb-6 w-60">
      <h3>Default</h3>
      <Dropdown options={people} value={people[0].id} />
    </div>
    <div className="mb-6 w-60">
      <h3>Disabled</h3>
      <Dropdown options={people} value={people[1].id} disabled={true} />
    </div>
    <div className="mb-6 w-60">
      <h3>Loading</h3>
      <Dropdown options={people} loading={true} />
    </div>
    <div className="mb-6 w-60">
      <h3>Error</h3>
      <Dropdown options={people} value={people[2].id} error="This is an error" />
    </div>
  </div>
);

export const AllVariations = AllTemplate.bind({});
export const Default = Template.bind({});
export const HasError = Template.bind({});

Default.args = {
  options: people,
  value: people[0].id,
};

HasError.args = {
  options: people,
  value: people[0].id,
  error: "This is an error",
};
