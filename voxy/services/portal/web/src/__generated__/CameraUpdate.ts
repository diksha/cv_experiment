/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL mutation operation: CameraUpdate
// ====================================================

export interface CameraUpdate_cameraUpdate_camera {
  __typename: "CameraType";
  /**
   * The ID of the object
   */
  id: string;
  /**
   * User friendly name displayed throughout apps.
   */
  name: string;
}

export interface CameraUpdate_cameraUpdate {
  __typename: "CameraUpdate";
  camera: CameraUpdate_cameraUpdate_camera | null;
}

export interface CameraUpdate {
  /**
   * Update an camera identified by its id
   */
  cameraUpdate: CameraUpdate_cameraUpdate | null;
}

export interface CameraUpdateVariables {
  cameraId: string;
  cameraName: string;
}
