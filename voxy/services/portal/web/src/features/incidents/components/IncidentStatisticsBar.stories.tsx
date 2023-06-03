import { Story } from "@storybook/react";
import { IncidentStatisticsBar } from "./IncidentStatisticsBar";

const config = {
  title: "Statistic/Bar",
  component: IncidentStatisticsBar,
};
export default config;

const Template: Story<React.ComponentProps<typeof IncidentStatisticsBar>> = (args) => (
  <div className="w-72 p-4">
    <IncidentStatisticsBar {...args} />
  </div>
);

export const Default = Template.bind({});
export const PercentageWidth = Template.bind({});
export const EmptyState = Template.bind({});

Default.args = {
  title: "Safety Vest",
  value: 121,
  barClassName: "bg-chart-safety-vest",
};

PercentageWidth.args = {
  title: "Hard Hat",
  value: 100,
  max: 300,
  barClassName: "bg-chart-hard-hat",
};

EmptyState.args = {
  title: "Spill",
  value: 0,
  barClassName: "bg-chart-spill",
};
