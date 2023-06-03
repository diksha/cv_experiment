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
import { useEffect } from "react";
import { useLazyQuery, useMutation } from "@apollo/client";

import { Drawer, INCIDENT_CREATE_SCENARIO } from "features/toolbox";
import { GET_INCIDENT_DETAILS } from "features/incidents";
import classNames from "classnames";
import { GetIncidentDetails, GetIncidentDetailsVariables } from "__generated__/GetIncidentDetails";
import { IncidentCreateScenario, IncidentCreateScenarioVariables } from "__generated__/IncidentCreateScenario";
import { ScenarioType } from "__generated__/globalTypes";

export function Scenarios(props: { incidentUuid: string }) {
  const [getIncidentDetails, { data }] = useLazyQuery<GetIncidentDetails, GetIncidentDetailsVariables>(
    GET_INCIDENT_DETAILS,
    {
      variables: {
        incidentUuid: props.incidentUuid,
      },
    }
  );

  const [incidentCreateScenario, { data: mutationData, loading: mutationLoading }] = useMutation<
    IncidentCreateScenario,
    IncidentCreateScenarioVariables
  >(INCIDENT_CREATE_SCENARIO);

  const handleCreateScenario = (scenarioType: ScenarioType) => {
    if (data?.incidentDetails?.id) {
      incidentCreateScenario({
        variables: { incidentId: data?.incidentDetails?.id, scenarioType },
      });
    }
  };

  const handleCreatePositiveScenario = () => {
    handleCreateScenario(ScenarioType.POSITIVE);
  };

  const handleCreateNegativeScenario = () => {
    handleCreateScenario(ScenarioType.NEGATIVE);
  };

  useEffect(() => {
    if (props.incidentUuid) {
      getIncidentDetails();
    }
  }, [getIncidentDetails, props.incidentUuid]);

  const buttonClasses = "w-full p-2 mb-2 rounded-md bg-gray-900";

  return (
    <Drawer name="Scenarios">
      {props.incidentUuid ? (
        <>
          <div>
            <button className={classNames(buttonClasses)} onClick={handleCreatePositiveScenario}>
              Create positive scenario
            </button>
          </div>
          <div>
            <button className={classNames(buttonClasses)} onClick={handleCreateNegativeScenario}>
              Create negative scenario
            </button>
          </div>
          <div>
            {mutationLoading ? "Creating scenario..." : null}
            {mutationData ? JSON.parse(mutationData.incidentCreateScenario?.incident?.data).scenario_s3_uri : null}
            Please run the ingest script https://buildkite.com/voxel/create-labeling-tasks to create labeling tasks for
            the scenario.
          </div>
        </>
      ) : (
        <div>Invalid incident UUID</div>
      )}
    </Drawer>
  );
}
