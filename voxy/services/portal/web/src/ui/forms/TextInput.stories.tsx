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
import { TextInput } from "./TextInput";

const config = {
  title: "Forms/TextInput",
  component: TextInput,
};
export default config;

const Template: Story<React.ComponentProps<typeof TextInput>> = (args) => <TextInput {...args} />;

export const Empty = Template.bind({});

Empty.args = {
  name: "firstName",
  label: "First name",
};

export const EmptyWithPlaceholder = Template.bind({});

EmptyWithPlaceholder.args = {
  name: "firstName",
  label: "First name",
  placeholder: "Enter your first name",
};
export const Populated = Template.bind({});

Populated.args = {
  name: "firstName",
  label: "First name",
  value: "Leonard",
};

export const Disabled = Template.bind({});

Disabled.args = {
  name: "firstName",
  label: "First name",
  disabled: true,
};

export const Error = Template.bind({});

Error.args = {
  name: "firstName",
  label: "First name",
  value: "%&^@",
  disabled: false,
  errors: ["This doesn't look like a valid name."],
};

export const MultipleErrors = Template.bind({});

MultipleErrors.args = {
  name: "firstName",
  label: "First name",
  value: "%&^@",
  disabled: false,
  errors: ["This doesn't look like a valid name.", "Unable to save your changes."],
};
