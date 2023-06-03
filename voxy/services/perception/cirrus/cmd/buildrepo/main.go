package main

import (
	"context"
	"flag"
	"fmt"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/spf13/afero"
	"github.com/voxel-ai/voxel/lib/utils/go/filesystem"
	cirruspb "github.com/voxel-ai/voxel/protos/perception/cirrus/v1"
	"github.com/voxel-ai/voxel/services/perception/cirrus/cmd/buildtriton"
	"log"
)

var configFile = flag.String("config", "", "repo config YAML file")
var repoPath = flag.String("repo-path", "", "repo path or s3")
var prune = flag.Bool("prune", true, "whether to prune extra models")
var models = flag.String("models", "services/perception/cirrus/config/models/", "whether to prune extra models")

func loadConfig() (*cirruspb.Repository, error) {
	dir := filesystem.NewDirectoryOperations(afero.NewOsFs())
	tritonConfig := buildtriton.NewTritonConfig(dir)
	modelRepoHelper := buildtriton.NewModelRepo(dir)
	modelMap, ensembleMap, err := tritonConfig.GetModelsFromConfig(*models)
	if err != nil {
		return nil, fmt.Errorf("failed to read model & ensemble defination files: %w", err)
	}
	graphModelRepoConfigMap, err := tritonConfig.GetProductionGraphModelRepoFromConfig(*configFile)
	if err != nil {
		return nil, fmt.Errorf("failed to graph model repo config defination files: %w", err)
	}
	graphModelRepoMap, err := tritonConfig.GetGraphConfigFromGraphModelRepo(graphModelRepoConfigMap)
	if err != nil {
		return nil, fmt.Errorf("failed to graph model repo config: %w", err)
	}
	repoModels, err := modelRepoHelper.GetModelsFromTritonConfig(modelMap, graphModelRepoMap)
	if err != nil {
		return nil, fmt.Errorf("failed to repo models : %w", err)
	}
	repoEnsembles, err := modelRepoHelper.GetEnsemblesFromTritonConfig(ensembleMap, graphModelRepoMap)
	if err != nil {
		return nil, fmt.Errorf("failed to repo models : %w", err)
	}
	cfg := &cirruspb.Repository{Models: repoModels, Ensembles: repoEnsembles}
	return cfg, nil
}

func main() {
	log.SetFlags(0)
	flag.Parse()

	ctx := context.Background()

	awsConfig, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		log.Fatal(err)
	}

	cfg, err := loadConfig()
	if err != nil {
		log.Fatal(err)
	}

	writer, err := NewRepoWriter(ctx, *repoPath)
	if err != nil {
		log.Fatal(err)
	}

	builder := &Builder{
		AWSConfig:  awsConfig,
		Config:     cfg,
		RepoWriter: writer,
	}

	if err := builder.Check(); err != nil {
		log.Fatal(err)
	}

	if err := builder.Build(ctx, *prune); err != nil {
		log.Fatal(err)
	}
}
