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
import React, { useEffect, useState } from "react";
import classNames from "classnames";
import { Card } from "ui";

interface TabProps {
  name: string;
  icon?: (props: React.ComponentProps<"svg">) => JSX.Element;
  children?: React.ReactNode;
  current?: boolean;
}

function Tab({ name }: TabProps) {
  return <div>{name}</div>;
}

interface TabCardProps {
  children?: React.ReactNode;
}

function TabCard({ children }: TabCardProps) {
  const [tabs, setTabs] = useState<TabProps[]>([]);

  useEffect(() => {
    const initTabs = React.Children.toArray(children)
      .filter((x: any) => React.isValidElement(x) && x.type === Tab)
      .map((x: any) => ({ ...x.props } as TabProps));

    // TODO: allow passing in default tab
    if (initTabs.length > 0) {
      initTabs[0].current = true;
    }
    setTabs(initTabs);
  }, [children]);

  const handleTabSelected = (name: string) => {
    setTabs(
      tabs.map((t: TabProps) => ({
        ...t,
        current: t.name === name,
      }))
    );
  };

  return (
    <Card noPadding>
      <div>
        <div className="sm:hidden p-4">
          <label htmlFor="tabs" className="sr-only">
            Select a tab
          </label>
          <select
            id="tabs"
            name="tabs"
            className="block w-full p-1 border focus:ring-indigo-500 focus:border-brand-blue-900 border-gray-300 rounded-md"
            onChange={(e) => handleTabSelected(e.target.value)}
            defaultValue={tabs.find((tab) => tab.current)?.name}
          >
            {tabs.map((tab) => (
              <option key={tab.name}>{tab.name}</option>
            ))}
          </select>
        </div>
        <div className="hidden sm:block">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8 pt-4 pr-4 pl-4" aria-label="Tabs">
              {tabs.map((tab) => (
                <span
                  key={tab.name}
                  onClick={() => handleTabSelected(tab.name!)}
                  className={classNames(
                    tab.current
                      ? "border-brand-blue-900 font-bold text-brand-blue-900 "
                      : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300",
                    "group inline-flex items-center py-4 px-1 border-b-2 text-sm cursor-pointer"
                  )}
                  aria-current={tab.current ? "page" : undefined}
                >
                  {tab.icon ? (
                    <tab.icon
                      className={classNames(
                        tab.current ? "text-indigo-500-ml-0.5" : "text-gray-400 group-hover:text-gray-500-ml-0.5",
                        "mr-2 h-5 w-5"
                      )}
                      aria-hidden="true"
                    />
                  ) : null}
                  <span>{tab.name}</span>
                </span>
              ))}
            </nav>
          </div>
        </div>
        {tabs.map((t: TabProps) =>
          t.current ? (
            <div key={t.name} className="p-4">
              {t.children}
            </div>
          ) : null
        )}
      </div>
    </Card>
  );
}

TabCard.Tab = Tab;

export default TabCard;
