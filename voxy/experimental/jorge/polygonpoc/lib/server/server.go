package server

import (
	"context"
	"fmt"

	_ "github.com/lib/pq"
	timestamppb "google.golang.org/protobuf/types/known/timestamppb"

	"github.com/voxel-ai/voxel/experimental/jorge/polygonpoc"
	"github.com/voxel-ai/voxel/experimental/jorge/polygonpoc/lib/portalquery"
)

type PolygonServer struct {
	polygonpoc.UnimplementedPolygonServiceServer
	queries *portalquery.Queries
}

// create a new gRPC server
func NewServer(cfg Config) *PolygonServer {
	return &PolygonServer{queries: portalquery.New(cfg.Database)}
}

func (srv *PolygonServer) GetCameras(ctx context.Context, in *polygonpoc.GetCamerasRequest) (*polygonpoc.GetCamerasResponse, error) {
	queriedCameras, err := srv.queries.GetCameras(context.Background())
	if err != nil {
		return nil, fmt.Errorf("failed to get cameras: %w", err)
	}

	camerasToReturn := &polygonpoc.GetCamerasResponse{}
	for _, camera := range queriedCameras {
		createdAtTimestamp := timestamppb.New(camera.CreatedAt)
		updatedAtTimestamp := timestamppb.New(camera.UpdatedAt)
		deletedAtTimestamp := timestamppb.New(camera.DeletedAt.Time)
		camerasToReturn.Cameras = append(camerasToReturn.Cameras, &polygonpoc.Camera{Id: camera.ID, CreatedAt: createdAtTimestamp, UpdatedAt: updatedAtTimestamp, DeletedAt: deletedAtTimestamp, Uuid: camera.Uuid, Name: camera.Name, OrganizationId: &camera.OrganizationID.Int64, ZoneId: &camera.ZoneID.Int64})
	}

	return camerasToReturn, nil

}
