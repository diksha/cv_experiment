package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"google.golang.org/protobuf/encoding/prototext"

	tritonpb "github.com/voxel-ai/voxel/experimental/jorge/modelrepo/third_party/triton"
)

func tritonModelFilename(mi ModelInfo) (string, error) {
	switch mi.Config.Platform {
	case "tensorrt_plan":
		return filepath.Join(mi.Name, "1/model.plan"), nil
	case "pytorch_libtorch":
		return filepath.Join(mi.Name, "1/model.pt"), nil
	case "ensemble":
		return filepath.Join(mi.Name, "1/ensemble"), nil
	default:
		return "", fmt.Errorf("unknown backend %q", mi.Config.Backend)
	}
}

// ModelInfo describes a triton model to be written
type ModelInfo struct {
	Name   string
	Config *tritonpb.ModelConfig
	Meta   map[string]string
}

// RepoFS is the interface required by RepoWriter to be able to write triton models
type RepoFS interface {
	WriteFile(name string, body io.Reader) error
	ListDirectory(prefix string) ([]string, error)
	ReadFile(name string) ([]byte, error)
	RemoveAll(name string) error
}

// RepoWriter writes models to a backend model repo filesystem according to the triton model repo spec
type RepoWriter struct {
	FS RepoFS
}

// WriteModelInfo writes a model file without writing the model data, allowing a models configuration to be
// updated without rewriting the whole model.
func (rw *RepoWriter) WriteModelInfo(mi ModelInfo) error {
	metaName := filepath.Join(mi.Name, "meta.json")
	metaBody, err := json.Marshal(mi.Meta)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	configName := filepath.Join(mi.Name, "config.pbtxt")
	configBody, err := prototext.MarshalOptions{Multiline: true}.Marshal(mi.Config)
	if err != nil {
		return fmt.Errorf("failed to marshal model config: %w", err)
	}

	if err := rw.FS.WriteFile(metaName, bytes.NewReader(metaBody)); err != nil {
		return fmt.Errorf("failed to write meta.json: %w", err)
	}

	if err := rw.FS.WriteFile(configName, bytes.NewReader(configBody)); err != nil {
		return fmt.Errorf("failed to write config.pbtxt: %w", err)
	}

	return nil
}

// WriteModel writes model data and model configuration to the model repository
func (rw *RepoWriter) WriteModel(mi ModelInfo, model io.Reader) error {
	if err := rw.WriteModelInfo(mi); err != nil {
		return fmt.Errorf("failed to write model info: %w", err)
	}

	modelFilename, err := tritonModelFilename(mi)
	if err != nil {
		return fmt.Errorf("unsupported model backend: %w", err)
	}

	if err := rw.FS.WriteFile(modelFilename, model); err != nil {
		return fmt.Errorf("failed to write model data: %w", err)
	}
	return nil
}

// ListModels returns the list of model names available in this model repo
func (rw *RepoWriter) ListModels() ([]string, error) {
	modelNames := make(map[string]bool)

	allFiles, err := rw.FS.ListDirectory("/")
	if err != nil {
		return nil, fmt.Errorf("failed to list model files: %w", err)
	}

	// sort model files to simplify the logic later
	for _, filename := range allFiles {
		// we want to be able to work with paths
		// that look like "model_name/1/model.pt" as well
		// as paths like "/model_name/1/model.pt"
		filename = strings.TrimPrefix(filename, "/")
		if len(filename) == 0 {
			continue
		}

		modelName := strings.Split(filename, "/")[0]
		modelNames[modelName] = true
	}

	var models []string
	for modelName := range modelNames {
		models = append(models, modelName)
	}

	return models, nil
}

func (rw *RepoWriter) readModelFiles(name string) (mi ModelInfo, files []string, err error) {
	allFiles, err := rw.FS.ListDirectory(name)
	if err != nil {
		return mi, nil, fmt.Errorf("failed to list model files for model %q: %w", name, err)
	}

	allFilesMap := make(map[string]bool)
	for _, filename := range allFiles {
		allFilesMap[filename] = true
	}

	metaName := filepath.Join(name, "meta.json")
	metaBody, err := rw.FS.ReadFile(metaName)
	if err != nil {
		return mi, nil, fmt.Errorf("failed to read meta.json: %w", err)
	}

	var meta map[string]string
	if err := json.Unmarshal(metaBody, &meta); err != nil {
		return mi, nil, fmt.Errorf("failed to unmarshal meta.json: %w", err)
	}

	configName := filepath.Join(name, "config.pbtxt")
	configBody, err := rw.FS.ReadFile(configName)
	if err != nil {
		return mi, nil, fmt.Errorf("failed to read config.pbtxt: %w", err)
	}

	var config tritonpb.ModelConfig
	if err := prototext.Unmarshal(configBody, &config); err != nil {
		return mi, nil, fmt.Errorf("failed to read config.pbtxt: %w", err)
	}

	delete(allFilesMap, metaName)
	delete(allFilesMap, configName)

	mi = ModelInfo{
		Name:   name,
		Config: &config,
		Meta:   meta,
	}

	modelFilename, err := tritonModelFilename(mi)
	if err != nil {
		return mi, nil, fmt.Errorf("unknown model filename: %w", err)
	}

	delete(allFilesMap, modelFilename)

	if len(allFilesMap) > 0 {
		unexpected := []string{}
		for filename := range allFilesMap {
			unexpected = append(unexpected, filename)
		}

		return mi, nil, fmt.Errorf("found unexpected files in model repo: %v", unexpected)
	}

	return mi, allFiles, nil
}

// ReadModel attempts to read model info for a model from the repository
func (rw *RepoWriter) ReadModel(name string) (ModelInfo, error) {
	mi, _, err := rw.readModelFiles(name)
	return mi, err
}

// DeleteModel deltes the passed in model from the model repository after first verifying that the model
// is valid and readable
func (rw *RepoWriter) DeleteModel(name string) error {
	_, _, err := rw.readModelFiles(name)
	if err != nil {
		return fmt.Errorf("error reading model before delete: %w", err)
	}

	if err := rw.FS.RemoveAll(name); err != nil {
		return fmt.Errorf("error removing model files for %q: %w", name, err)
	}

	return nil
}

// HasModel returns true if the specified model exists in the repo, false if not, and an error if
// there is an error checking for model files
func (rw *RepoWriter) HasModel(name string) (bool, error) {
	files, err := rw.FS.ListDirectory(name)
	if err != nil {
		return false, fmt.Errorf("failed to list model files: %w", err)
	}
	return len(files) > 0, nil
}

// NewRepoWriter attempts to construct a valid repo writer for the specified repo path.
// Currently only s3 urls and filesystem paths are supported.
func NewRepoWriter(repoPath string) (*RepoWriter, error) {
	var err error
	var repofs RepoFS

	if strings.HasPrefix(repoPath, "s3://") {
		repofs, err = NewS3RepoFS(context.TODO(), repoPath)
	} else {
		repofs, err = newOSRepoFS(repoPath)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to initialize repo backend: %w", err)
	}

	return &RepoWriter{repofs}, nil
}
