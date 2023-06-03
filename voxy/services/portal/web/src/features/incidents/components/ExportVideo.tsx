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
import { Menu, Transition } from "@headlessui/react";
import React, { Fragment } from "react";
import styles from "./ExportVideo.module.css";
import { RiVideoDownloadFill } from "react-icons/ri";
import { Spinner } from "ui";
import { useMutation } from "@apollo/client";
import { EXPORT_VIDEO } from "features/incidents";
import { IncidentExportVideo, IncidentExportVideoVariables } from "__generated__/IncidentExportVideo";

export function ExportVideo(props: { incidentId: string }) {
  const [exportVideo, { loading }] = useMutation<IncidentExportVideo, IncidentExportVideoVariables>(EXPORT_VIDEO);

  const handleExport = async (labeled: boolean) => {
    const response = await exportVideo({
      variables: {
        incidentId: props.incidentId,
        labeled,
      },
    });
    const downloadUrl = response?.data?.incidentExportVideo?.downloadUrl;
    if (downloadUrl) {
      window.open(downloadUrl, "_blank");
    } else {
      // TODO: display a toast notification
      console.error("Failed to export video", response);
    }
  };

  return (
    <div className="relative inline-block text-left">
      <Menu as="div">
        <div>
          {loading ? (
            <Spinner />
          ) : (
            <Menu.Button className={styles.download} title="Export video">
              <RiVideoDownloadFill />
            </Menu.Button>
          )}
        </div>
        <Transition
          as={Fragment}
          enter="transition ease-out duration-100"
          enterFrom="transform opacity-0 scale-95"
          enterTo="transform opacity-100 scale-100"
          leave="transition ease-in duration-75"
          leaveFrom="transform opacity-100 scale-100"
          leaveTo="transform opacity-0 scale-95"
        >
          <Menu.Items className="absolute right-0 w-56 mt-2 origin-top-right bg-white divide-y divide-gray-100 rounded-md shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none z-50">
            <div className="px-1 py-1">
              <Menu.Item>
                {({ active }) => (
                  <button
                    data-ui-key="button-export-video"
                    className={`${active && "bg-gray-100"} group flex rounded-md w-full px-2 py-2 text-sm`}
                    onClick={() => handleExport(false)}
                  >
                    Export video
                  </button>
                )}
              </Menu.Item>
              <Menu.Item>
                {({ active }) => (
                  <button
                    data-ui-key="button-export-video-with-labels"
                    className={`${active && "bg-gray-100"} text-black group flex rounded-md w-full px-2 py-2 text-sm`}
                    onClick={() => handleExport(true)}
                  >
                    Export video with labels
                  </button>
                )}
              </Menu.Item>
            </div>
          </Menu.Items>
        </Transition>
      </Menu>
    </div>
  );
}
