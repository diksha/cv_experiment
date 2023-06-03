/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetSiteCameras
// ====================================================

export interface GetSiteCameras_zone_cameras_edges_node_incidentTypes {
  __typename: "CameraIncidentType";
  id: string;
  name: string;
  key: string;
  backgroundColor: string;
}

export interface GetSiteCameras_zone_cameras_edges_node {
  __typename: "CameraType";
  /**
   * The ID of the object
   */
  id: string;
  /**
   * User friendly name displayed throughout apps.
   */
  name: string;
  thumbnailUrl: string | null;
  incidentTypes: GetSiteCameras_zone_cameras_edges_node_incidentTypes[];
}

export interface GetSiteCameras_zone_cameras_edges {
  __typename: "CameraTypeEdge";
  /**
   * The item at the end of the edge
   */
  node: GetSiteCameras_zone_cameras_edges_node | null;
}

export interface GetSiteCameras_zone_cameras {
  __typename: "CameraTypeConnection";
  /**
   * Contains the nodes in this connection.
   */
  edges: (GetSiteCameras_zone_cameras_edges | null)[];
}

export interface GetSiteCameras_zone {
  __typename: "ZoneType";
  /**
   * The ID of the object
   */
  id: string;
  cameras: GetSiteCameras_zone_cameras | null;
}

export interface GetSiteCameras {
  zone: GetSiteCameras_zone | null;
}

export interface GetSiteCamerasVariables {
  zoneId: string;
}
