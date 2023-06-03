/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

import { ScenarioType } from "./globalTypes";

// ====================================================
// GraphQL mutation operation: IncidentCreateScenario
// ====================================================

export interface IncidentCreateScenario_incidentCreateScenario_incident {
  __typename: "IncidentType";
  /**
   * The ID of the object
   */
  id: string;
  pk: number;
  data: any | null;
}

export interface IncidentCreateScenario_incidentCreateScenario {
  __typename: "IncidentCreateScenario";
  incident: IncidentCreateScenario_incidentCreateScenario_incident | null;
}

export interface IncidentCreateScenario {
  incidentCreateScenario: IncidentCreateScenario_incidentCreateScenario | null;
}

export interface IncidentCreateScenarioVariables {
  incidentId: string;
  scenarioType: ScenarioType;
}
