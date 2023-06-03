/* eslint-disable */
"use strict";

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
const Node = /** @class */ (function () {
  function Node(value) {
    this.value = value;
    this.height = 1;
    this.left = null;
    this.right = null;
  }
  return Node;
})();

// TODO(PERCEPTION-984) - Remove duplicate code services/portal/web/src/shared/dataStructures/AVLTree.ts
/**
 * Class representing an AVL tree.
 *
 * https://en.wikipedia.org/wiki/AVL_tree
 */
const AVLTree = /** @class */ (function () {
  function AVLTree() {
    this.root = null;
  }
  AVLTree.prototype.getNodeHeight = function (node) {
    if (node === null) {
      return 0;
    }
    return node.height;
  };
  AVLTree.prototype.rightRotate = function (y) {
    // TODO: will x ever be null? If so, handle it.
    const x = y.left;
    const T2 = x.right;
    x.right = y;
    y.left = T2;
    y.height = Math.max(this.getNodeHeight(y.left), this.getNodeHeight(y.right)) + 1;
    x.height = Math.max(this.getNodeHeight(x.left), this.getNodeHeight(x.right)) + 1;
    return x;
  };
  AVLTree.prototype.leftRotate = function (x) {
    // TODO: will y ever be null? If so, handle it.
    const y = x.right;
    const T2 = y.left;
    y.left = x;
    x.right = T2;
    x.height = Math.max(this.getNodeHeight(x.left), this.getNodeHeight(x.right)) + 1;
    y.height = Math.max(this.getNodeHeight(y.left), this.getNodeHeight(y.right)) + 1;
    return y;
  };
  AVLTree.prototype.getBalanceFactor = function (node) {
    if (node === null) {
      return 0;
    }
    return this.getNodeHeight(node.left) - this.getNodeHeight(node.right);
  };
  AVLTree.prototype.insertNodeHelper = function (node, value) {
    // find the position and insert the node
    if (node === null) {
      return new Node(value);
    }
    if (value < node.value) {
      node.left = this.insertNodeHelper(node.left, value);
    } else if (value > node.value) {
      node.right = this.insertNodeHelper(node.right, value);
    } else {
      return node;
    }
    // update the balance factor of each node
    // and, balance the tree
    node.height = 1 + Math.max(this.getNodeHeight(node.left), this.getNodeHeight(node.right));
    const balanceFactor = this.getBalanceFactor(node);
    if (balanceFactor > 1) {
      if (value < node.value.item) {
        return this.rightRotate(node);
      }
      if (value > node.left.value) {
        node.left = this.leftRotate(node.left);
        return this.rightRotate(node);
      }
    }
    if (balanceFactor < -1) {
      if (value > node.right.value) {
        return this.leftRotate(node);
      }
      if (value < node.right.value) {
        node.right = this.rightRotate(node.right);
        return this.leftRotate(node);
      }
    }
    return node;
  };
  AVLTree.prototype.insert = function (data) {
    this.root = this.insertNodeHelper(this.root, data);
  };
  AVLTree.prototype.nodeWithMinimumValue = function (node) {
    let current = node;
    while (current.left !== null) {
      current = current.left;
    }
    return current;
  };
  AVLTree.prototype.deleteNodeHelper = function (root, value) {
    // find the node to be deleted and remove it
    if (root === null) {
      return root;
    }
    if (value < root.value) {
      root.left = this.deleteNodeHelper(root.left, value);
    } else if (value > root.value) {
      root.right = this.deleteNodeHelper(root.right, value);
    } else if (root.left === null || root.right === null) {
      var temp = null;
      if (temp === root.left) {
        temp = root.right;
      } else {
        temp = root.left;
      }
      if (temp === null) {
        temp = root;
        root = null;
      } else {
        root = temp;
      }
    } else {
      var temp = this.nodeWithMinimumValue(root.right);
      root.value = temp.value;
      root.right = this.deleteNodeHelper(root.right, temp.value);
    }
    if (root === null) {
      return root;
    }
    // Update the balance factor of each node and balance the tree
    root.height = Math.max(this.getNodeHeight(root.left), this.getNodeHeight(root.right)) + 1;
    const balanceFactor = this.getBalanceFactor(root);
    if (balanceFactor > 1) {
      if (this.getBalanceFactor(root.left) >= 0) {
        return this.rightRotate(root);
      }
      root.left = this.leftRotate(root.left);
      return this.rightRotate(root);
    }
    if (balanceFactor < -1) {
      if (this.getBalanceFactor(root.right) <= 0) {
        return this.leftRotate(root);
      }
      root.right = this.rightRotate(root.right);
      return this.leftRotate(root);
    }
    return root;
  };
  AVLTree.prototype.delete = function (value) {
    this.root = this.deleteNodeHelper(this.root, value);
  };
  /**
   * Finds the nearest value present in the tree.
   * @param value the value to find.
   * @param strategy the rounding strategy to use.
   * @param node the tree node to evaluate (typically only used recursively).
   * @returns the nearest value if found, otherwise null.
   */
  AVLTree.prototype.findNearest = function (value, strategy) {
    let _a;
    // Perfect match, rounding strategy doesn't apply, return this value
    if (((_a = this.root) === null || _a === void 0 ? void 0 : _a.value) === value) {
      return value;
    }
    switch (strategy) {
      case "floor":
        return this.findNearestFloor(this.root, value);
      case "ceiling":
        return this.findNearestCeiling(this.root, value);
      default:
        // Should never reach this point
        return null;
    }
  };
  /**
   * Find nearest value less-than-or-equal-to provided value.
   */
  AVLTree.prototype.findNearestFloor = function (node, value) {
    if (!node) {
      return null;
    }
    // Nearest value must be a descendant of left child
    if (value < node.value) {
      return this.findNearestFloor(node.left, value);
    }
    // Nearest value will be either current node or a descendant of right child
    if (value > node.value) {
      const nearestRightChildData = this.findNearestFloor(node.right, value);
      if (nearestRightChildData !== null) {
        return Math.max(node.value, nearestRightChildData);
      }
      return node.value;
    }
    // Current node matches the provided value
    return value;
  };
  /**
   * Find nearest value greater-than-or-equal-to provided value.
   */
  AVLTree.prototype.findNearestCeiling = function (node, value) {
    if (!node) {
      return null;
    }
    // Nearest value must be a descendant of right child
    if (node.value < value) {
      return this.findNearestCeiling(node.right, value);
    }
    // Nearest value will be either current node or a descendant of right child
    if (value < node.value) {
      const nearestLeftChildData = this.findNearestCeiling(node.left, value);
      if (nearestLeftChildData !== null) {
        return Math.min(node.value, nearestLeftChildData);
      }
      return node.value;
    }
    // Current node matches the provided value
    return value;
  };
  AVLTree.prototype.traverseInorder = function (node) {
    return this.traverse(this.root, "inorder");
  };
  AVLTree.prototype.traversePreorder = function (node) {
    return this.traverse(this.root, "preorder");
  };
  AVLTree.prototype.traversePostorder = function (node) {
    return this.traverse(this.root, "postorder");
  };
  AVLTree.prototype.traverse = function (node, order) {
    const output = [];
    if (node) {
      switch (order) {
        case "inorder":
          output.push.apply(output, this.traverse(node.left, order));
          output.push(node.value);
          output.push.apply(output, this.traverse(node.right, order));
          break;
        case "preorder":
          output.push(node.value);
          output.push.apply(output, this.traverse(node.left, order));
          output.push.apply(output, this.traverse(node.right, order));
          break;
        case "postorder":
          output.push.apply(output, this.traverse(node.left, order));
          output.push.apply(output, this.traverse(node.right, order));
          output.push(node.value);
          break;
        default:
          // Should never get here
          break;
      }
    }
    return output;
  };
  AVLTree.prototype.printInorder = function () {
    this.traverseInorder(this.root).forEach(function (val) {
      return console.log(val);
    });
  };
  AVLTree.prototype.printPreorder = function () {
    this.traversePreorder(this.root).forEach(function (val) {
      return console.log(val);
    });
  };
  AVLTree.prototype.printPostorder = function () {
    this.traversePostorder(this.root).forEach(function (val) {
      return console.log(val);
    });
  };
  return AVLTree;
})();
