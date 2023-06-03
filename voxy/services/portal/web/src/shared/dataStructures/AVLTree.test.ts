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
import { AVLTree } from "shared/dataStructures/AVLTree";

it("produces expected outputs", () => {
  const testCases = [
    {
      inputs: [5, 4, 3, 2, 1],
      inorderOutput: [1, 2, 3, 4, 5],
    },
    {
      inputs: [3, 2, 4, 1, 5],
      inorderOutput: [1, 2, 3, 4, 5],
    },
    {
      inputs: [1, 2, 3, 4, 5],
      inorderOutput: [1, 2, 3, 4, 5],
    },
    {
      inputs: [1, 5, 2, 4, 3, 3, 4, 2, 5, 1],
      inorderOutput: [1, 2, 3, 4, 5],
    },
    {
      inputs: [1, 10, 2, 9, 3, 8, 4, 7, 5, 6],
      inorderOutput: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    },
  ];

  for (const testCase of testCases) {
    const tree = new AVLTree();
    testCase.inputs.forEach((value) => tree.insert(value));
    expect(tree.traverseInorder()).toEqual(testCase.inorderOutput);
  }
});
