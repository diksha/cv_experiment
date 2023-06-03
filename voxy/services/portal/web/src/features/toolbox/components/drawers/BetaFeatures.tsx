/*
 * Copyright 2020-2022 Voxel Labs, Inc.
 * All rights reserved.
 *
 * This document may not be reproduced, republished, distributed, transmitted,
 * displayed, broadcast or otherwise exploited in any manner without the express
 * prior written permission of Voxel Labs, Inc. The receipt or possession of this
 * document does not convey any rights to reproduce, disclose, or distribute its
 * contents, or to manufacture, use, or sell anything that it may describe, in
 * whole or in part.
 */
import { Link } from "react-router-dom";
import { TestTube, CaretRight } from "phosphor-react";
import { Drawer } from "features/toolbox";

const features = [
  { name: "Analytics V2", path: "/beta/analytics" },
  { name: "Account V2", path: "/beta/account" },
];

function Content(props: { onClick?: () => void }) {
  return (
    <>
      {features.length > 0 ? (
        <div className="flex flex-col gap-4">
          {features.map((feature) => (
            <Link
              key={feature.path}
              to={feature.path}
              className="flex p-2 items-center rounded-md w-full bg-brand-gray-400 hover:opacity-80"
              onClick={props.onClick}
            >
              <div className="flex-grow">{feature.name}</div>
              <div>
                <CaretRight className="h-5 w-5" />
              </div>
            </Link>
          ))}
        </div>
      ) : (
        <div className="text-center py-4">No beta features available</div>
      )}
    </>
  );
}

export function BetaFeatures(props: { onFeatureClicked?: () => void }) {
  return (
    <Drawer name="Beta features" icon={<TestTube className="h-6 w-6 text-brand-green-300" />}>
      <Content onClick={props.onFeatureClicked} />
    </Drawer>
  );
}
