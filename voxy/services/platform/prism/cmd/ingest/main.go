// Package main is the entry point for processing incidents to be ingested into Prism
package main

import (
	"context"
	"os"

	"github.com/rs/zerolog/log"

	"github.com/aws/aws-lambda-go/lambda"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideo"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideoarchivedmedia"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/cristalhq/aconfig"

	"github.com/voxel-ai/voxel/services/platform/prism/lib/fragarchive"
	"github.com/voxel-ai/voxel/services/platform/prism/lib/incident"
	"github.com/voxel-ai/voxel/services/platform/prism/lib/ingest"
)

type config struct {
	FragmentArchiveBucket string `required:"true" usage:"name of the S3 bucket in which fragments should be archived"`
	IncidentInputPath     string `usage:"path to a .json file containing incident data (for running locally)"`
	LocalRun              bool   `usage:"specify when running this outside of the lambda function"`
}

func loadConfig() config {
	appConfig := config{}
	configLoader := aconfig.LoaderFor(&appConfig, aconfig.Config{EnvPrefix: "PRISM"})
	if err := configLoader.Load(); err != nil {
		log.Fatal().Err(err).Msg("failed to load application configuration")
	}

	if appConfig.LocalRun && appConfig.IncidentInputPath == "" {
		log.Fatal().Msg("must define incident input path for local run")
	}

	if !appConfig.LocalRun && appConfig.IncidentInputPath != "" {
		log.Fatal().Msg("must not define an incident input path for non-local runs")
	}

	log.Info().
		Str("FragmentArchiveBucket", appConfig.FragmentArchiveBucket).
		Str("IncidentInputPath", appConfig.IncidentInputPath).
		Bool("LocalRun", appConfig.LocalRun).
		Msg("Loaded config")

	return appConfig
}

func main() {
	appConfig := loadConfig()

	ctx := context.Background()
	awsConfig, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		log.Fatal().Err(err).Msg("failed to load AWS config")
	}

	archiveClient := fragarchive.New(
		s3.NewFromConfig(awsConfig),
		appConfig.FragmentArchiveBucket,
	)
	kvsClient := kvClientWithEndpointCaching{
		innerClient:   kinesisvideo.NewFromConfig(awsConfig),
		endpointCache: globalEndpointCacheManager,
	}
	kvamClient := kinesisvideoarchivedmedia.NewFromConfig(awsConfig)

	ingestClient := ingest.Client{
		ArchiveClient: archiveClient,
		KVClient:      kvsClient,
		KVAMClient:    kvamClient,
	}

	if appConfig.LocalRun {
		ctx := context.Background()
		fileBytes, err := os.ReadFile(appConfig.IncidentInputPath)
		if err != nil {
			log.Fatal().Err(err).Msgf("failed to read file %v", appConfig.IncidentInputPath)
		}

		incident, err := incident.Unmarshal(string(fileBytes))
		if err != nil {
			log.Fatal().Err(err).Str("input", string(fileBytes)).Msgf("failed to unmarshall incident from contents of %v", appConfig.IncidentInputPath)
		}

		err = ingestClient.HandleIncident(ctx, incident)
		if err != nil {
			log.Fatal().Err(err).Msg("failed to ingest incident")
		}
	} else {
		lambda.Start(ingestClient.HandleSQSEvent)
	}
}
