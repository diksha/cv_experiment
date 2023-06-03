package buildtriton_test

import (
	"github.com/stretchr/testify/assert"
	"github.com/voxel-ai/voxel/lib/utils/go/filesystem/filesystemfakes"
	cirruspb "github.com/voxel-ai/voxel/protos/perception/cirrus/v1"
	graphconfigpb "github.com/voxel-ai/voxel/protos/perception/graph_config/v1"
	"github.com/voxel-ai/voxel/services/perception/cirrus/cmd/buildtriton"
	"testing"
)

func TestGetModelsFromTritonConfig(t *testing.T) {
	modelMap := map[string]*cirruspb.Model{
		"model1": {
			ArtifactModelPaths: []string{"artifact_path_1", "artifact_path_3"},
		},
		"model2": {
			ArtifactModelPaths: []string{"artifact_path_2", "artifact_path_4"},
		},
	}
	dtPath := "artifact_path_2"
	graphConfig := &graphconfigpb.GraphConfig{
		Perception: &graphconfigpb.PerceptionConfig{
			DetectorTracker: &graphconfigpb.DetectorTrackerPerceptionConfig{ModelPath: &dtPath},
		},
	}
	graphConfigMap := map[string]*graphconfigpb.GraphConfig{
		"graph1": graphConfig}

	fakeDir := new(filesystemfakes.FakeDirectoryOperationsInterface)
	setupTest(fakeDir)
	modelRepoConstructor := buildtriton.NewModelRepo(fakeDir)

	modelList, err := modelRepoConstructor.GetModelsFromTritonConfig(modelMap, graphConfigMap)

	assert.NoError(t, err)
	assert.Len(t, modelList, 1)
	result := modelList[0].ArtifactModelPaths[0]
	expected := "artifact_path_2"
	assert.Equal(t, result, expected)
}
