// material-ui
import { createTheme } from "@mui/material/styles";
import { PaletteMode } from "@mui/material";

// assets
import defaultThemeColors from "assets/scss/_defaultThemeColors.module.scss";

// types
import { ColorProps } from "types";

// ==============================|| DEFAULT THEME - PALETTE  ||============================== //

const Palette = (navType: PaletteMode, presetColor: string) => {
  let colors: ColorProps;
  switch (presetColor) {
    // Additional themes can be mapped here...
    case "default":
    default:
      colors = defaultThemeColors;
  }

  return createTheme({
    palette: {
      mode: navType,
      common: {
        white: colors.paper,
        black: colors.darkPaper,
      },
      primary: {
        light: navType === "dark" ? colors.darkPrimaryLight : colors.primaryLight,
        main: navType === "dark" ? colors.darkPrimaryMain : colors.primaryMain,
        dark: navType === "dark" ? colors.darkPrimaryDark : colors.primaryDark,
        100: navType === "dark" ? colors.darkPrimary100 : colors.primary100,
        200: navType === "dark" ? colors.darkPrimary200 : colors.primary200,
        300: navType === "dark" ? colors.darkPrimary300 : colors.primary300,
        400: navType === "dark" ? colors.darkPrimary400 : colors.primary400,
        500: navType === "dark" ? colors.darkPrimary500 : colors.primary500,
        600: navType === "dark" ? colors.darkPrimary600 : colors.primary600,
        700: navType === "dark" ? colors.darkPrimary700 : colors.primary700,
        800: navType === "dark" ? colors.darkPrimary800 : colors.primary800,
        900: navType === "dark" ? colors.darkPrimary900 : colors.primary900,
      },
      secondary: {
        light: navType === "dark" ? colors.darkSecondaryLight : colors.secondaryLight,
        main: navType === "dark" ? colors.darkSecondaryMain : colors.secondaryMain,
        dark: navType === "dark" ? colors.darkSecondaryDark : colors.secondaryDark,
        100: navType === "dark" ? colors.darkSecondary100 : colors.secondary100,
        200: navType === "dark" ? colors.darkSecondary200 : colors.secondary200,
        300: navType === "dark" ? colors.darkSecondary300 : colors.secondary300,
        400: navType === "dark" ? colors.darkSecondary400 : colors.secondary400,
        500: navType === "dark" ? colors.darkSecondary500 : colors.secondary500,
        600: navType === "dark" ? colors.darkSecondary600 : colors.secondary600,
        700: navType === "dark" ? colors.darkSecondary700 : colors.secondary700,
        800: navType === "dark" ? colors.darkSecondary800 : colors.secondary800,
        900: navType === "dark" ? colors.darkSecondary900 : colors.secondary900,
      },
      error: {
        light: colors.errorLight,
        main: colors.errorMain,
        dark: colors.errorDark,
        600: colors.error600,
      },
      orange: {
        light: colors.orangeLight,
        main: colors.orangeMain,
        dark: colors.orangeDark,
      },
      warning: {
        light: colors.warningLight,
        main: colors.warningMain,
        dark: colors.warningDark,
      },
      success: {
        light: colors.successLight,
        main: colors.successMain,
        dark: colors.successDark,
        contrastText: colors.paper,
        200: colors.success200,
        600: colors.success600,
      },
      grey: {
        100: colors.grey100,
        200: colors.grey200,
        300: colors.grey300,
        400: colors.grey400,
        500: colors.grey500,
        600: colors.grey600,
        700: colors.grey700,
        800: colors.grey800,
        900: colors.grey900,
      },
      dark: {
        light: colors.darkTextPrimary,
        main: colors.darkLevel1,
        dark: colors.darkLevel2,
        800: colors.darkBackground,
        900: colors.darkPaper,
      },
      text: {
        primary: navType === "dark" ? colors.darkTextPrimary : colors.grey900,
        secondary: navType === "dark" ? colors.darkTextSecondary : colors.grey700,
        dark: navType === "dark" ? colors.darkTextPrimary : colors.grey900,
        hint: colors.grey100,
      },
      divider: navType === "dark" ? colors.darkTextPrimary : colors.grey200,
      background: {
        paper: navType === "dark" ? colors.darkLevel2 : colors.paper,
        default: "#edf0f4",
      },
    },
  });
};

export default Palette;
