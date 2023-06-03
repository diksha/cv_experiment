import { gql } from "@apollo/client";
export const CAMERA_UPDATE = gql`
  mutation CameraUpdate($cameraId: ID!, $cameraName: String!) {
    cameraUpdate(cameraId: $cameraId, cameraName: $cameraName) {
      camera {
        id
        name
      }
    }
  }
`;
