import { Story } from "@storybook/react";
import { ActionPill, PillActions } from "./ActionPill";

const config = {
  title: "Incidents/ActionPill",
  component: ActionPill,
};
export default config;

const Template: Story<React.ComponentProps<typeof ActionPill>> = (args) => (
  <div className="w-1/2 h-72">
    <ActionPill {...args} />
  </div>
);

export const Alerted = Template.bind({});
Alerted.args = {
  type: PillActions.ALERTED,
};

export const HighPriority = Template.bind({});
HighPriority.args = {
  type: PillActions.HIGH_PRIORITY,
};

export const MediumPriority = Template.bind({});
MediumPriority.args = {
  type: PillActions.MEDIUM_PRIORITY,
};

export const LowPriority = Template.bind({});
LowPriority.args = {
  type: PillActions.LOW_PRIORITY,
};

export const Assigned = Template.bind({});
Assigned.args = {
  type: PillActions.ASSIGNED,
};

export const Resolved = Template.bind({});
Resolved.args = {
  type: PillActions.RESOLVED,
};
