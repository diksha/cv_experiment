package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"google.golang.org/protobuf/encoding/protojson"
	"sigs.k8s.io/yaml"

	"github.com/aws/aws-sdk-go-v2/config"

	"github.com/voxel-ai/voxel/experimental/jorge/modelrepo"
)

var configFile = flag.String("config", "", "repo config YAML file")
var repoPath = flag.String("repo-path", "", "repo path or s3")
var prune = flag.Bool("prune", true, "whether to prune extra models")

func loadConfig() (*modelrepo.Repository, error) {
	data, err := os.ReadFile(*configFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	rawjson, err := yaml.YAMLToJSON(data)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal config JSON: %w", err)
	}

	cfg := &modelrepo.Repository{}
	if err := protojson.Unmarshal(rawjson, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config protobuf: %w", err)
	}

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

	writer, err := NewRepoWriter(*repoPath)
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
