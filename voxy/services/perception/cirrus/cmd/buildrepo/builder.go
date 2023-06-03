// buildrepo is a program that builds triton model repos based on an input yaml file
package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/bazelbuild/rules_go/go/runfiles"
	"google.golang.org/protobuf/proto"

	cirruspb "github.com/voxel-ai/voxel/protos/perception/cirrus/v1"
	tritonpb "github.com/voxel-ai/voxel/protos/third_party/triton"
)

const (
	modelNameSalt   = "_do_not_manually_generate_triton_model_names_"
	artifactURLKey  = "ArtifactURL"
	artifactPathKey = "ArtifactPath"
	artifactSHAKey  = "ArtifactSHA256"
	primaryModelKey = "PrimaryModel"
)

// ArtifactMeta is the schema for artifact metdata stored in the voxel artifact system
type ArtifactMeta struct {
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
	URL    string `json:"url"`
}

// Builder is a model repo builder that is capable of constructin a model repo either locally or in aws
type Builder struct {
	AWSConfig  aws.Config
	Config     *cirruspb.Repository
	RepoPath   string
	RepoWriter *RepoWriter
}

func (b *Builder) fetchArtifactMeta(artifactPath string) (ArtifactMeta, error) {
	dirname, _ := path.Split(artifactPath)
	metaRunfilesPath := path.Join(dirname, "meta.json")

	runf, err := runfiles.New()
	if err != nil {
		return ArtifactMeta{}, fmt.Errorf("failed to find bazel runfiles: %w", err)
	}

	metaPath, err := runf.Rlocation(metaRunfilesPath)
	if err != nil {
		return ArtifactMeta{}, fmt.Errorf("failed to find artifact metdata: %w", err)
	}

	rawjson, err := os.ReadFile(metaPath)
	if err != nil {
		return ArtifactMeta{}, fmt.Errorf("failed to read artifact metadata file: %w", err)
	}

	meta := ArtifactMeta{}
	if err := json.Unmarshal(rawjson, &meta); err != nil {
		return ArtifactMeta{}, fmt.Errorf("failed to parse artifact metdata file: %w", err)
	}

	return meta, nil
}

func (b *Builder) fetchArtifact(ctx context.Context, meta ArtifactMeta) ([]byte, error) {
	bucket, key, err := parseS3URL(meta.URL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse artifact url: %w", err)
	}

	resp, err := s3.NewFromConfig(b.AWSConfig).GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return nil, fmt.Errorf("s3 get object error: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	buf := &bytes.Buffer{}

	if _, err := io.Copy(buf, resp.Body); err != nil {
		return nil, fmt.Errorf("s3 get object download error: %w", err)
	}

	sumBytes := sha256.Sum256(buf.Bytes())
	sum := hex.EncodeToString(sumBytes[:])
	if sum != meta.SHA256 {
		return nil, fmt.Errorf("artifact checksum mismatch got %s expected %s", sum, meta.SHA256)
	}

	return buf.Bytes(), nil
}

func (b *Builder) fetchModelFromArtifact(ctx context.Context, modelPath string, meta ArtifactMeta) ([]byte, error) {
	modelPathSpl := strings.Split(modelPath, "/")
	if len(modelPathSpl) < 2 {
		return nil, fmt.Errorf("invalid model path %v for model fetch", modelPath)
	}

	filename := filepath.Join(modelPathSpl[1:]...)

	data, err := b.fetchArtifact(ctx, meta)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch artifact: %w", err)
	}

	gzr, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to read artifact gzip data: %w", err)
	}

	artifact := tar.NewReader(gzr)
	for {
		hdr, err := artifact.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("error reading artifact tar data: %w", err)
		}

		if filepath.Clean(hdr.Name) != filepath.Clean(filename) {
			continue
		}

		data, err := io.ReadAll(artifact)
		if err != nil {
			return nil, fmt.Errorf("error reading model data from archive: %w", err)
		}

		return data, nil
	}

	return nil, fmt.Errorf("failed to find artifact %q in artifact %q", filename, meta.URL)
}

var modelNameCharsRegex = regexp.MustCompile(`[^a-zA-Z0-9_]`)

func (b *Builder) modelName(artifactModelPath string) string {
	sumBytes := sha256.Sum256([]byte(artifactModelPath + modelNameSalt))
	sum := hex.EncodeToString(sumBytes[:])[0:6]
	sanitizedName := modelNameCharsRegex.ReplaceAllString(artifactModelPath, "_")
	return fmt.Sprintf("%s_%s", sanitizedName, sum)
}

func (b *Builder) cloneModel(modelName string, baseConfig *tritonpb.ModelConfig) (*tritonpb.ModelConfig, error) {
	config := proto.Clone(baseConfig).(*tritonpb.ModelConfig)
	if config.Name != "" {
		return nil, fmt.Errorf("model config must not specify Name")
	}
	return config, nil
}

func (b *Builder) generateModelWarmup(modelName string, baseConfig *tritonpb.ModelConfig) (*tritonpb.ModelConfig, error) {
	config, err := b.cloneModel(modelName, baseConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to clone model config: %w", err)
	}
	if len(config.ModelWarmup) == 0 {
		warmup := &tritonpb.ModelWarmup{}
		warmup.Name = "randomized"
		warmup.Count = 10

		warmup.BatchSize = uint32(config.MaxBatchSize)
		if warmup.BatchSize <= 0 {
			warmup.BatchSize = 1
		}

		warmup.Inputs = make(map[string]*tritonpb.ModelWarmup_Input)
		for _, inputConfig := range config.Input {
			warmup.Inputs[inputConfig.Name] = &tritonpb.ModelWarmup_Input{
				DataType:      inputConfig.DataType,
				Dims:          inputConfig.Dims,
				InputDataType: &tritonpb.ModelWarmup_Input_RandomData{RandomData: true},
			}
		}

		config.ModelWarmup = append(config.ModelWarmup, warmup)
	}

	return config, nil
}

// Build constructs the model repo according to the configuration passed in, optionally pruning excess models
func (b *Builder) Build(ctx context.Context, prune bool) error {
	log.Printf("running build with prune=%v", prune)
	existingModels, err := b.RepoWriter.ListModels()
	if err != nil {
		return fmt.Errorf("failed to list existing repo models: %w", err)
	}

	existingModelSet := make(map[string]bool)
	for _, modelName := range existingModels {
		log.Printf("Found existing model: %v", modelName)
		existingModelSet[modelName] = true
	}

	log.Printf("Found %d existing models", len(existingModelSet))

	for _, model := range b.Config.Models {
		for _, artifactModelPath := range model.ArtifactModelPaths {
			modelName := b.modelName(artifactModelPath)
			log.Printf("Updating model %v", modelName)
			artifactMeta, err := b.fetchArtifactMeta(artifactModelPath)
			if err != nil {
				return fmt.Errorf("failed to get artifact meta: %w", err)
			}

			modelConfig, err := b.cloneModel(modelName, model.Config)
			if err != nil {
				return fmt.Errorf("failed to clone model config: %w", err)
			}

			if !model.DisableWarmupGeneration {
				modelConfig, err = b.generateModelWarmup(modelName, modelConfig)
				if err != nil {
					return fmt.Errorf("failed to generate model warmup: %w", err)
				}
			}

			if err := b.updateModel(ctx, modelName, modelConfig, artifactModelPath, artifactMeta); err != nil {
				return fmt.Errorf("failed to update model: %w", err)
			}
			delete(existingModelSet, modelName)
		}
	}

	for _, ensemble := range b.Config.Ensembles {
		for _, artifactModelPath := range ensemble.ArtifactModelPaths {
			modelName := b.modelName(artifactModelPath)
			ensembleName := modelName + "_ensemble"
			log.Printf("Updating ensemble %v", ensembleName)
			if err := b.updateEnsemble(ctx, ensembleName, modelName, ensemble); err != nil {
				return fmt.Errorf("failed to update ensemble: %w", err)
			}
			delete(existingModelSet, ensembleName)
		}
	}

	log.Printf("Finished updating models")

	if !prune {
		return nil
	}

	log.Printf("Pruning excess models")
	for modelName := range existingModelSet {
		log.Printf("Deleting model: %v", modelName)
		if err := b.RepoWriter.DeleteModel(modelName); err != nil {
			return fmt.Errorf("failed to delete mode: %w", err)
		}
	}

	log.Printf("Finished pruning models")

	return nil
}

func (b *Builder) updateModel(ctx context.Context, modelName string, modelConfig *tritonpb.ModelConfig, modelPath string, artifactMeta ArtifactMeta) error {
	modelInfo := ModelInfo{
		Name:   modelName,
		Config: modelConfig,
		Meta: map[string]string{
			artifactSHAKey:  artifactMeta.SHA256,
			artifactURLKey:  artifactMeta.URL,
			artifactPathKey: modelPath,
		},
	}

	hasModel, err := b.RepoWriter.HasModel(modelName)
	if err != nil {
		return fmt.Errorf("failed to check if model exists: %w", err)
	}

	var existingModelMeta map[string]string
	if hasModel {
		existingModelInfo, err := b.RepoWriter.ReadModel(modelName)
		if err != nil {
			return fmt.Errorf("failed to read model metadata: %w", err)
		}
		existingModelMeta = existingModelInfo.Meta
	}

	if !reflect.DeepEqual(modelInfo.Meta, existingModelMeta) {
		modelData, err := b.fetchModelFromArtifact(ctx, modelPath, artifactMeta)
		if err != nil {
			return fmt.Errorf("failed to download model data: %w", err)
		}

		if err := b.RepoWriter.WriteModel(modelInfo, bytes.NewReader(modelData)); err != nil {
			return fmt.Errorf("failed to write model: %w", err)
		}
	} else {
		if err := b.RepoWriter.WriteModelInfo(modelInfo); err != nil {
			return fmt.Errorf("failed to write model info: %w", err)
		}
	}
	return nil
}

func (b *Builder) updateEnsemble(ctx context.Context, ensembleName, modelName string, ensemble *cirruspb.Ensemble) error {
	ensembleConfig, err := b.cloneModel(ensembleName, ensemble.Config)
	if err != nil {
		return fmt.Errorf("failed to generate model config: %w", err)
	}

	foundPrimaryModel := false
	for _, step := range ensembleConfig.GetEnsembleScheduling().Step {
		if step.ModelName == ensemble.PrimaryModelName {
			foundPrimaryModel = true
			step.ModelName = modelName
		} else {
			step.ModelName = b.modelName(step.ModelName)
		}
	}
	if !foundPrimaryModel {
		return fmt.Errorf("failed to find primary model name %q in model ensemble %q", ensemble.PrimaryModelName, ensembleName)
	}

	modelInfo := ModelInfo{
		Name:   ensembleName,
		Config: ensembleConfig,
	}

	if err := b.RepoWriter.WriteModel(modelInfo, bytes.NewReader(nil)); err != nil {
		return fmt.Errorf("failed to write model info: %w", err)
	}
	return nil
}

// Check validates the passed in build configuration
func (b *Builder) Check() error {
	for _, model := range b.Config.Models {
		if model.Config == nil {
			return fmt.Errorf("must specify model configuration for all models")
		}

		if len(model.ArtifactModelPaths) == 0 {
			return fmt.Errorf("must specify at least one entry in artifact_model_paths")
		}
	}

	for _, ensemble := range b.Config.Ensembles {
		if len(ensemble.PrimaryModelName) == 0 {
			return fmt.Errorf("must specify primary_model_name for all ensembles")
		}

		if ensemble.Config.Platform != "ensemble" {
			return fmt.Errorf("platform must be \"ensemble\" for all ensembles")
		}

		foundPrimaryModel := false
		for _, step := range ensemble.Config.GetEnsembleScheduling().Step {
			if step.ModelName == ensemble.PrimaryModelName {
				foundPrimaryModel = true
			}
		}
		if !foundPrimaryModel {
			return fmt.Errorf("no step with model_name %q found in ensemble", ensemble.PrimaryModelName)
		}
	}
	return nil
}
