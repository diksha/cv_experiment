import React from "react";
import { addDecorator } from "@storybook/react";
import { MemoryRouter } from "react-router-dom";
import "../src/assets/css/app.base.css";

export const parameters = {
  actions: { argTypesRegex: "^on[A-Z].*" },
  controls: {
    matchers: {
      color: /(background|color)$/i,
      date: /Date$/,
    },
  },
};

// Required to make react-router components work within Storybook
addDecorator((story) => {
  return <MemoryRouter initialEntries={["/"]}>{story()}</MemoryRouter>;
});
