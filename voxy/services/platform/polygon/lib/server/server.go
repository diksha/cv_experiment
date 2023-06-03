package server

import (
	"bytes"
	"context"
	"fmt"
	"strings"

	"sigs.k8s.io/yaml"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"

	_ "github.com/lib/pq"
	timestamppb "google.golang.org/protobuf/types/known/timestamppb"

	"google.golang.org/protobuf/encoding/protojson"

	graph_config "github.com/voxel-ai/voxel/protos/perception/graph_config/v1"
	polygon "github.com/voxel-ai/voxel/protos/platform/polygon/v1"
	"github.com/voxel-ai/voxel/services/platform/polygon/lib/portalquery"
)

const reasonMetadata = "x-amz-meta-reason"

// PolygonServer for Polygon service
type PolygonServer struct {
	polygon.UnimplementedPolygonServiceServer
	queries  *portalquery.Queries
	config   Config
	s3Client S3API
}

// S3API has PutObject and GetObject to be implemented by the server
type S3API interface {
	PutObject(context.Context, *s3.PutObjectInput, ...func(*s3.Options)) (*s3.PutObjectOutput, error)
	GetObject(context.Context, *s3.GetObjectInput, ...func(*s3.Options)) (*s3.GetObjectOutput, error)
}

var _ S3API = (*s3.Client)(nil)

// NewServer creates a new gRPC server
func NewServer(s3Client S3API, cfg Config) (*PolygonServer, error) {
	return &PolygonServer{queries: portalquery.New(cfg.Database), config: cfg, s3Client: s3Client}, nil
}

func (srv *PolygonServer) GetCameras(ctx context.Context, in *polygon.GetCamerasRequest) (*polygon.GetCamerasResponse, error) {
	queriedCameras, err := srv.queries.GetCameras(context.Background())
	if err != nil {
		return nil, fmt.Errorf("failed to get cameras: %w", err)
	}

	camerasToReturn := &polygon.GetCamerasResponse{}
	for _, camera := range queriedCameras {
		createdAtTimestamp := timestamppb.New(camera.CreatedAt)
		updatedAtTimestamp := timestamppb.New(camera.UpdatedAt)
		deletedAtTimestamp := timestamppb.New(camera.DeletedAt.Time)
		camerasToReturn.Cameras = append(camerasToReturn.Cameras, &polygon.Camera{Id: camera.ID, CreatedAt: createdAtTimestamp, UpdatedAt: updatedAtTimestamp, DeletedAt: deletedAtTimestamp, Uuid: camera.Uuid, Name: camera.Name, OrganizationId: &camera.OrganizationID.Int64, ZoneId: &camera.ZoneID.Int64})
	}

	return camerasToReturn, nil
}

// PutGraphConfigYAML accepts a request containing info on the bucket, the key, the graph config, and reason and puts that info to S3
func (srv *PolygonServer) PutGraphConfigYAML(ctx context.Context, request *polygon.PutGraphConfigYAMLRequest) (*polygon.PutGraphConfigYAMLResponse, error) {
	graphConfigInput := request.GraphConfig
	graphConfigFormat := graph_config.GraphConfig{}

	rawJSON, err := yaml.YAMLToJSON([]byte(graphConfigInput))
	if err != nil {
		return nil, fmt.Errorf("failed to marshal config JSON: %w", err)
	}

	// check to make sure that the yaml file fits in the graph config proto schema
	if err := protojson.Unmarshal(rawJSON, &graphConfigFormat); err != nil {
		return nil, fmt.Errorf("failed to parse config protobuf: %w", err)
	}

	_, err = srv.s3Client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:   aws.String(srv.config.Bucket),
		Key:      aws.String(request.CameraUuid),
		Body:     strings.NewReader(graphConfigInput),
		Metadata: map[string]string{reasonMetadata: request.Reason},
	})

	if err != nil {
		return nil, fmt.Errorf("failed to push Polygon override to s3: %w", err)
	}

	response := &polygon.PutGraphConfigYAMLResponse{}
	return response, nil
}

// GetGraphConfigYAML accepts a request containing info on the bucket and the key and returns the Polygon override
func (srv *PolygonServer) GetGraphConfigYAML(ctx context.Context, request *polygon.GetGraphConfigYAMLRequest) (*polygon.GetGraphConfigYAMLResponse, error) {
	s3Response, err := srv.s3Client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(srv.config.Bucket),
		Key:    aws.String(request.CameraUuid),
	})
	defer func() {
		_ = s3Response.Body.Close()
	}()
	if err != nil {
		return nil, fmt.Errorf("failed to get Polygon override from s3: %w", err)
	}

	graphConfigBuffer := new(bytes.Buffer)
	_, err = graphConfigBuffer.ReadFrom(s3Response.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read the request's response body: %w", err)
	}

	response := &polygon.GetGraphConfigYAMLResponse{
		GraphConfig:  graphConfigBuffer.String(),
		LastModified: timestamppb.New(*s3Response.LastModified),
		Reason:       s3Response.Metadata[reasonMetadata],
	}

	return response, nil
}
