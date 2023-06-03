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
export const randomColorString = () => {
  return "#" + ((Math.random() * 0xffffff) << 0).toString(16);
};

/**
 * Generates array of hex color codes for video annotations.
 * Intended to be consumed as a stack with the hardcoded values being
 * the preferred colors and randomly generated colors serving as backup
 * if there are more actors than hardcoded colors.
 */
export const generateColorStack = () => {
  const preferredColors = [
    "#E74C3C",
    "#8E44AD",
    "#5DADE2",
    "#48C9B0",
    "#F4D03F",
    "#E67E22",
    "#F5B7B1",
    "#D2B4DE",
    "#85C1E9",
    "#82E0AA",
    "#F7DC6F",
    "#EDBB99",
  ];

  const randomColors = [...Array(25)].map(() => randomColorString());
  return preferredColors.concat(randomColors);
};
