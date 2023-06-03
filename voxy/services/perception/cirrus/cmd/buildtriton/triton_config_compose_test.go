package buildtriton_test

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/encoding/protojson"
	"sigs.k8s.io/yaml"

	"github.com/voxel-ai/voxel/lib/utils/go/filesystem/filesystemfakes"
	cirruspb "github.com/voxel-ai/voxel/protos/perception/cirrus/v1"
	"github.com/voxel-ai/voxel/services/perception/cirrus/cmd/buildtriton"
)

func setupTest(fakeDirectory *filesystemfakes.FakeDirectoryOperationsInterface) {
	fakeDirectory.IsDirectoryStub = func(string) (bool, error) {
		return false, nil
	}
	fakeDirectory.IsFileStub = func(string) (bool, error) {
		return true, nil
	}

	fakeDirectory.GetFilePathsStub = func(dirPath string, extension string) ([]string, error) {
		if extension != ".yaml" {
			return nil, errors.New("invalid arguments")
		}
		switch dirPath {
		case "/path/to/yaml/files/services":
			return []string{"/path/to/yaml/files/services/service1.yaml", "/path/to/yaml/files/services/service2.yaml"}, nil
		case "/path/to/yaml/files/malformed_services":
			return []string{"/path/to/yaml/files/services/service3.yaml"}, nil
		case "/path/to/yaml/config/services":
			return []string{"/path/to/yaml/config/services/production_graph_model_repo.yaml"}, nil
		default:
			return nil, errors.New("invalid arguments")
		}
	}

	fakeDirectory.ReadFileStub = func(path string) ([]byte, error) {
		switch path {
		case "/path/to/yaml/files/services/service1.yaml":
			data, _ := yaml.JSONToYAML([]byte("{\n  \"model\": {\n    \"artifact_model_paths\": [\n      \"path1\",\n      \"path2\"\n    ]\n  }\n}"))
			return data, nil
		case "/path/to/yaml/files/services/service2.yaml":
			data, _ := yaml.JSONToYAML([]byte("{\n  \"ensemble\": {\n    \"primary_model_name\": \"PrimaryEnsemble\",\n    \"artifact_model_paths\": [\n      \"path1\",\n      \"path2\"\n    ]\n  }\n}"))
			return data, nil
		case "/path/to/yaml/files/services/service3.yaml":
			data, _ := yaml.JSONToYAML([]byte("{\n  \"model\": {\n    \"artifact_model_paths\": [\n      \"path1\",\n      \"path2\"\n    ]\n  },\n  \"ensemble\": {\n    \"primary_model_name\": \"PrimaryEnsemble\",\n    \"artifact_model_paths\": [\n      \"path1\",\n      \"path2\"\n    ]\n  }\n}"))
			return data, nil
		case "/path/to/yaml/config/services/production_graph_model_repo.yaml":
			data, _ := yaml.JSONToYAML([]byte("{\n \"name\":\"test\" , \"graph_config_paths\": [\n    \"my-repo\"\n  ]\n}"))
			return data, nil
		case "services/cha.yaml":
			data, _ := yaml.JSONToYAML([]byte("{\n  \"perception\": {\n    \"detector_tracker\": {\n      \"model_path\": \"/path/to/detector_tracker/model\"\n    }\n  }\n}"))
			return data, nil
		default:
			return nil, fmt.Errorf("invalid file path: %s", path)
		}
	}
}

func TestTritonConfig_GetModelsFromConfig_WellFormedYAML(t *testing.T) {
	fakeDir := new(filesystemfakes.FakeDirectoryOperationsInterface)
	setupTest(fakeDir)
	tritonCfg := buildtriton.NewTritonConfig(fakeDir)
	var testModel cirruspb.Model
	testModel.ArtifactModelPaths = []string{"path1", "path2"}
	expectedModels := map[string]*cirruspb.Model{
		"services/service1.yaml": &testModel,
	}
	var testEnsemble cirruspb.Ensemble
	testEnsemble.ArtifactModelPaths = []string{"path1", "path2"}
	testEnsemble.PrimaryModelName = "PrimaryEnsemble"
	expectedEnsembles := map[string]*cirruspb.Ensemble{
		"services/service2.yaml": &testEnsemble,
	}

	models, ensembles, err := tritonCfg.GetModelsFromConfig("/path/to/yaml/files/services")
	if err != nil {
		t.Fatalf("failed to get models from config: %v", err)
	}

	if len(models) != len(expectedModels) {
		t.Fatalf("expected %d models, but got %d", len(expectedModels), len(models))
	}
	if len(ensembles) != len(expectedEnsembles) {
		t.Fatalf("expected %d ensemble, but got %d", len(expectedEnsembles), len(ensembles))
	}

	for path, expectedModel := range expectedModels {
		model, ok := models[path]
		if !ok {
			t.Fatalf("model not found for path %s", path)
		}

		expectedJSON, _ := protojson.Marshal(expectedModel)
		json, _ := protojson.Marshal(model)

		if !bytes.Equal(expectedJSON, json) {
			t.Fatalf("expected model %s, but got %s", string(expectedJSON), string(json))
		}
	}
	for path, expectedEnsemble := range expectedEnsembles {
		ensemble, ok := ensembles[path]
		if !ok {
			t.Fatalf("ensembles not found for path %s", path)
		}

		expectedJSON, _ := protojson.Marshal(expectedEnsemble)
		json, _ := protojson.Marshal(ensemble)

		if !bytes.Equal(expectedJSON, json) {
			t.Fatalf("expected ensembles %s, but got %s", string(expectedJSON), string(json))
		}
	}
}

func TestTritonConfig_GetModelsFromConfig_MalformedYAML(t *testing.T) {
	fakeDir := new(filesystemfakes.FakeDirectoryOperationsInterface)
	setupTest(fakeDir)
	tritonCfg := buildtriton.NewTritonConfig(fakeDir)

	models, ensembles, err := tritonCfg.GetModelsFromConfig("/path/to/yaml/files/malformed_services")
	assert.Nil(t, models)
	assert.Nil(t, ensembles)

	assert.True(t, strings.Contains(err.Error(), "failed to convert the yaml file to obj"))
}

func TestTritonConfig_GetProductionGraphModelRepoFromConfig(t *testing.T) {
	fakeDir := new(filesystemfakes.FakeDirectoryOperationsInterface)
	setupTest(fakeDir)
	tritonCfg := buildtriton.NewTritonConfig(fakeDir)

	var testGraphModelConfig cirruspb.ProductionGraphModelRepo
	testGraphModelConfig.GraphConfigPaths = []string{"my-repo"}
	testGraphModelConfig.Name = "test"

	expectedRepos := map[string]*cirruspb.ProductionGraphModelRepo{
		"services/production_graph_model_repo.yaml": &testGraphModelConfig,
	}

	repos, err := tritonCfg.GetProductionGraphModelRepoFromConfig("/path/to/yaml/config/services")
	if err != nil {
		t.Fatalf("failed to get production graph model repos from config: %v", err)
	}
	if len(repos) != len(expectedRepos) {
		t.Fatalf("expected %d repos, but got %d", len(expectedRepos), len(repos))
	}

	for path, expectedRepo := range expectedRepos {
		repo, ok := repos[path]
		if !ok {
			t.Fatalf("repo not found for path %s", path)
		}

		expectedJSON, _ := protojson.Marshal(expectedRepo)
		json, _ := protojson.Marshal(repo)

		if !bytes.Equal(expectedJSON, json) {
			t.Fatalf("expected repo %s, but got %s", string(expectedJSON), string(json))
		}
	}
}

func TestTritonConfig_GetGraphConfigFromGraphModelRepo(t *testing.T) {
	fakeDir := new(filesystemfakes.FakeDirectoryOperationsInterface)
	setupTest(fakeDir)
	tritonCfg := buildtriton.NewTritonConfig(fakeDir)

	var testGraphModelConfig cirruspb.ProductionGraphModelRepo
	testGraphModelConfig.GraphConfigPaths = []string{"services/cha.yaml"}
	input := map[string]*cirruspb.ProductionGraphModelRepo{
		"services/cha.yaml": &testGraphModelConfig,
	}

	result, err := tritonCfg.GetGraphConfigFromGraphModelRepo(input)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	resultPath := result["services/cha.yaml"].Perception.DetectorTracker.ModelPath
	assert.Equal(t, *resultPath, "/path/to/detector_tracker/model")
}

func TestTritonConfig_GetGraphConfigFromGraphModelRepo_DuplicateFilePath_ShouldError(t *testing.T) {
	fakeDir := new(filesystemfakes.FakeDirectoryOperationsInterface)
	setupTest(fakeDir)
	tritonCfg := buildtriton.NewTritonConfig(fakeDir)

	var testGraphModelConfig cirruspb.ProductionGraphModelRepo
	testGraphModelConfig.GraphConfigPaths = []string{"services/cha.yaml"}
	input := map[string]*cirruspb.ProductionGraphModelRepo{
		"services/cha.yaml":  &testGraphModelConfig,
		"services1/cha.yaml": &testGraphModelConfig,
	}

	result, err := tritonCfg.GetGraphConfigFromGraphModelRepo(input)
	expectedError := errors.New("duplicate path found in config map: services/cha.yaml")
	assert.Equal(t, err, expectedError)
	assert.Nil(t, result)
}
