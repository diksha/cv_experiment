package main

import (
	"context"
	"os"

	"github.com/aws/aws-lambda-go/lambda"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/secretsmanager"
	"github.com/rs/zerolog"

	"github.com/voxel-ai/voxel/lib/utils/csrsigner"
)

func main() {
	ctx := context.Background()
	logger := zerolog.New(os.Stderr).With().Timestamp().Logger()
	ctx = logger.WithContext(ctx)

	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		logger.Fatal().Err(err).Msg("failed to load aws config")
	}

	sm := secretsmanager.NewFromConfig(cfg)
	signer, err := csrsigner.LoadFromSecretsManager(ctx, sm, os.Getenv("DEVCERT_SECRET_ARN"))
	if err != nil {
		logger.Fatal().Err(err).Msg("failed to load csr signer")
	}

	h := &Handler{
		Signer: signer,
	}

	lambda.StartWithOptions(h.Handle, lambda.WithContext(ctx))
}
