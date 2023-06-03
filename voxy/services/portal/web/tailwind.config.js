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
module.exports = {
  corePlugins: {
    // The default preflight styles conflicted with Material UI, so we
    // added a copy to `assets/css/tailwind.preflight.css` with patches
    // to improve Tailwind + Material UI interop.
    preflight: false,
  },
  content: ["./src/**/*.{js,jsx,ts,tsx}", "./public/index.html"],
  theme: {
    fontFamily: {
      sans: ["Karla", "sans-serif"],
      karla: ["Karla", "sans-serif"],
      epilogue: ["Epilogue", "sans-serif"],
      fira: ["Fira Mono", "sans-serif"],
    },
    minWidth: {
      180: "180px",
      450: "450px",
    },
    extend: {
      fontSize: {
        xxs: ".6rem",
      },
      colors: {
        // NOTE. If you update/add/remove these colors, please apply the same changes to the colors in app.base.css
        "brand-primary": {
          100: "#d8e1f5",
          200: "#b4c4eb",
          300: "#7f92c5",
          400: "#4e5c8b",
          500: "#1a223f",
          600: "#131936",
          700: "#0d122d",
          800: "#080c24",
          900: "#04081e",
        },
        "brand-gray": {
          900: "#00000A",
          800: "#010114",
          700: "#01011F",
          600: "#020229",
          500: "#020233",
          400: "#35355C",
          300: "#676785",
          200: "#9A9AAD",
          100: "#CCCCD6",
          "050": "#E6E6EB",
          "000": "#FFFFFF",
        },
        "brand-purple": {
          900: "#140927",
          800: "#28114E",
          700: "#3B1A74",
          600: "#4F229B",
          500: "#632BC2",
          400: "#8255CE",
          300: "#A180DA",
          200: "#C1AAE7",
          100: "#E0D5F3",
        },
        "brand-blue": {
          900: "#0B0D31",
          800: "#161A62",
          700: "#212694",
          600: "#2C33C5",
          500: "#3740F6",
          400: "#5F66F8",
          300: "#878CFA",
          200: "#AFB3FB",
          100: "#D7D9FD",
        },
        "brand-red": {
          900: "#27100C",
          800: "#4E1F18",
          700: "#762F24",
          600: "#9D3E30",
          500: "#C44E3C",
          400: "#D07163",
          300: "#DC958A",
          200: "#E7B8B1",
          100: "#F3DCD8",
        },
        "brand-orange": {
          900: "#2D170E",
          800: "#5B2D1B",
          700: "#884429",
          600: "#B65A36",
          500: "#E37144",
          400: "#E98D69",
          300: "#EEAA8F",
          200: "#F4C6B4",
          100: "#F9E3DA",
        },
        "brand-yellow": {
          900: "#302B05",
          800: "#61550A",
          700: "#918010",
          600: "#C2AB15",
          500: "#F2D51A",
          400: "#F5DE48",
          300: "#F7E676",
          200: "#FAEEA3",
          100: "#FDF7D1",
        },
        "brand-green": {
          900: "#142B1B",
          800: "#275735",
          700: "#3B8250",
          600: "#4EAE6A",
          500: "#62D985",
          400: "#81E19D",
          300: "#A1E8B6",
          200: "#C0F0CE",
          100: "#E0F7E7",
        },
        chart: {
          spill: "#5cc7fa",
          "open-door": "#1d91c0",
          "door-violation": "#225ea8",
          piggyback: "#4d004b",
          "no-stop": "#810f7c",
          "intersection-speeding": "#8c6bb1",
          "end-aisle-speeding": "#df65b0",
          parking: "#c51b7d",
          "no-ped-zone": "#8c510a",
          posture: "#bf812d",
          "safety-vest": "#dfc27d",
          "hard-hat": "#fee08b",
          ppe: "#d9f0a3",
        },
      },
      transitionProperty: {
        height: "height",
        width: "width",
        "max-height": "max-height",
      },
      animation: {
        "spin-slow": "spin 2s linear infinite",
      },
      boxShadow: ["active"],
      backgroundColor: ["active"],
      textColor: ["active"],
      borderWidth: {
        3: "3px",
      },
      spacing: {
        4.5: "1.125rem",
      },
    },
  },
  variants: {
    extend: {
      borderRadius: ["first", "last"],
      borderWidth: ["first", "last"],
      backgroundColor: ["even", "odd"],
    },
  },
  plugins: [require("@tailwindcss/forms"), require("@tailwindcss/line-clamp")],
};
