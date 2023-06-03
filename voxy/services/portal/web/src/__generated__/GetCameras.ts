/* tslint:disable */
/* eslint-disable */
// @generated
// This file was automatically generated and should not be edited.

// ====================================================
// GraphQL query operation: GetCameras
// ====================================================

export interface GetCameras_cameras {
  __typename: "CameraType";
  uuid: string;
  /**
   * User friendly name displayed throughout apps.
   */
  name: string;
}

export interface GetCameras {
  cameras: (GetCameras_cameras | null)[] | null;
}
