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

export function UnlockIcon(props: { className: string }) {
  return (
    <svg
      className={props.className}
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M13 17H11M6 21H18C19.1046 21 20 20.1046 20 19V13C20 11.8954 19.1046 11 18 11H6C4.89543 11 4 11.8954 4 13V19C4 20.1046 4.89543 21 6 21Z"
        stroke="#111827"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M12.8652 3.74109C11.0556 2.47398 8.56142 2.91377 7.29431 4.72339C6.0272 6.53301 6.46699 9.0272 8.27661 10.2943L9.09576 10.8679"
        stroke="#111827"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}
