package server

import (
	"context"
	"database/sql"
	"os"

	"fmt"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/bazelbuild/rules_go/go/runfiles"
	embeddedpostgres "github.com/fergusstrange/embedded-postgres"
	"github.com/jmoiron/sqlx"
	_ "github.com/lib/pq"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	polygon "github.com/voxel-ai/voxel/protos/platform/polygon/v1"
	"github.com/voxel-ai/voxel/services/platform/polygon/lib/portalquery"
)

func Test_ManyTestsAgainstOneDatabase(t *testing.T) {
	const sampleID = 12345
	const sampleUUID = "0000-0000-0000-0000"
	const sampleName = "test name"
	const sampleKey = "test key"
	const sampleTimezone = "test timezone"

	// compiler gives errors when making these into const
	sampleOrganizationID := int64(2)
	sampleZoneID := int64(3)
	sampleTime := time.Unix(1677087214, 0)

	runfile, err := runfiles.New()
	require.NoError(t, err, "runfiles.New() to set up Runfiles type should succeed")
	portalPath, err := runfile.Rlocation("voxel/services/platform/polygon/lib/portalquery/portal.sql")
	require.NoError(t, err, "runfile.Rlocation() to get portal.sql schema should succeed")
	portalSQL, err := os.ReadFile(portalPath)
	require.NoError(t, err, "reading of portal.sql schema should succeed")

	database := embeddedpostgres.NewDatabase()
	err = database.Start()
	require.NoError(t, err, "starting the database should succeed")

	defer func() {
		err = database.Stop()
		require.NoError(t, err, "stopping the database should succeed")
	}()

	db, err := connect()
	require.NoError(t, err, "connecting to database should succeed")

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	queries := portalquery.New(db)

	_, err = db.Exec("CREATE USER voxelapp") // needed since the tables include the statement "OWNER TO voxelapp"
	require.NoError(t, err, "executing statement to create user voxelapp should succeed")

	_, err = db.Exec(string(portalSQL))
	require.NoError(t, err, "executing statments to import portal.sql schema should succeed")

	insertAPIOrganizationParams := portalquery.InsertApiOrganizationParams{
		ID:        int64(sampleOrganizationID),
		CreatedAt: sampleTime,
		UpdatedAt: sampleTime,
		DeletedAt: sql.NullTime{Time: sampleTime, Valid: true},
		Name:      sampleName,
		Key:       sampleKey,
		IsSandbox: true,
		Timezone:  sampleTimezone,
	}

	insertZoneParams := portalquery.InsertZoneParams{
		ID:             sampleZoneID, // must use sampleZoneID here to satisfy foreign key constraint
		CreatedAt:      sampleTime,
		UpdatedAt:      sampleTime,
		DeletedAt:      sql.NullTime{Time: sampleTime, Valid: true},
		Name:           sampleName,
		OrganizationID: int64(sampleOrganizationID),
		ZoneType:       "test ztype", // 10 char max
		ParentZoneID:   sql.NullInt64{Int64: int64(sampleZoneID), Valid: true},
		Key:            sampleKey,
		Timezone:       sql.NullString{String: sampleTimezone, Valid: true},
		Active:         true,
	}

	insertCameraParams := portalquery.InsertCameraParams{
		ID:             int64(sampleID),
		CreatedAt:      sampleTime,
		UpdatedAt:      sampleTime,
		DeletedAt:      sql.NullTime{Time: sampleTime, Valid: true},
		Uuid:           sampleUUID,
		Name:           sampleName,
		OrganizationID: sql.NullInt64{Int64: int64(sampleOrganizationID), Valid: true},
		ZoneID:         sql.NullInt64{Int64: int64(sampleZoneID), Valid: true},
	}

	_, err = queries.InsertApiOrganization(ctx, insertAPIOrganizationParams)
	require.NoError(t, err, "insert on api org table should succeed")

	_, err = queries.InsertZone(ctx, insertZoneParams)
	require.NoError(t, err, "insert on zone table should succeed")

	_, err = queries.InsertCamera(ctx, insertCameraParams)
	require.NoError(t, err, "insert on camera table should succeed")

	awsConfig, err := config.LoadDefaultConfig(ctx)
	require.NoError(t, err, "loading default config should succeed")
	s3Client := s3.NewFromConfig(awsConfig)
	srv, err := NewServer(s3Client, Config{Server: Server{}, Database: db.DB})
	require.NoError(t, err, "making a new server should succeed")
	tests := []func(t *testing.T){
		func(t *testing.T) {
			sampleOrganizationID64 := int64(sampleOrganizationID)
			expectedCameras := [1]*polygon.Camera{{
				Id:             sampleID,
				CreatedAt:      timestamppb.New(sampleTime),
				UpdatedAt:      timestamppb.New(sampleTime),
				DeletedAt:      timestamppb.New(sampleTime),
				Uuid:           sampleUUID,
				Name:           sampleName,
				OrganizationId: &sampleOrganizationID64,
				ZoneId:         &sampleZoneID,
			}}

			resp, err := srv.GetCameras(ctx, &polygon.GetCamerasRequest{})
			require.NoError(t, err, "failed to GetCameras")

			actualCameras := resp.GetCameras()

			if len(expectedCameras) != len(actualCameras) || !proto.Equal(expectedCameras[0], actualCameras[0]) {
				t.Fatalf("expected %+v did not match actual %+v", expectedCameras, actualCameras)
			}
		},
		// more tests can be added here and they will be run without needing to rebuild the db each time
	}

	for testNumber, test := range tests {
		t.Run(fmt.Sprintf("%d", testNumber), test)
	}
}

func connect() (*sqlx.DB, error) {
	db, err := sqlx.Connect("postgres", "host=localhost port=5432 user=postgres password=postgres dbname=postgres sslmode=disable")
	return db, err
}
