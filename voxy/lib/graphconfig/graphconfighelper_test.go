package graphconfig_test

import (
	"fmt"
	"github.com/hashicorp/go-set"
	"github.com/spf13/afero"
	"github.com/stretchr/testify/assert"
	"github.com/voxel-ai/voxel/lib/graphconfig"
	"github.com/voxel-ai/voxel/lib/utils/go/filesystem"
	graphconfigpb "github.com/voxel-ai/voxel/protos/perception/graph_config/v1"
	"testing"
)

func TestGetAllModelPathFromGraphConfig(t *testing.T) {
	// Define model paths
	dtPath := "dt_path"
	doorPath := "door_path"
	hatPath := "hat_path"
	liftPath := "lift_path"
	posePath := "pose_path"
	reachPath := "reach_path"
	vestPath := "vest_path"
	spillPath := "spill_path"
	carryPath := "carry_path"

	// Setup the graph config
	graphConfig := &graphconfigpb.GraphConfig{
		Perception: &graphconfigpb.PerceptionConfig{
			DetectorTracker:       &graphconfigpb.DetectorTrackerPerceptionConfig{ModelPath: &dtPath},
			DoorClassifier:        &graphconfigpb.DoorClassifierPerceptionConfig{ModelPath: &doorPath},
			HatClassifier:         &graphconfigpb.HatClassifierPerceptionConfig{ModelPath: &hatPath},
			LiftClassifier:        &graphconfigpb.LiftClassifierPerceptionConfig{ModelPath: &liftPath},
			Pose:                  &graphconfigpb.PosePerceptionConfig{ModelPath: &posePath},
			ReachClassifier:       &graphconfigpb.ReachClassifierPerceptionConfig{ModelPath: &reachPath},
			VestClassifier:        &graphconfigpb.VestClassifierPerceptionConfig{ModelPath: &vestPath},
			Spill:                 &graphconfigpb.SpillPerceptionConfig{ModelPath: &spillPath},
			CarryObjectClassifier: &graphconfigpb.CarryObjectPerceptionConfig{ModelPath: &carryPath},
		},
	}

	expectedModelPaths := set.From[string]([]string{dtPath, doorPath, hatPath, liftPath, posePath, reachPath, vestPath, spillPath, carryPath})

	memMapFs := afero.NewMemMapFs()
	directoryOperations := filesystem.NewDirectoryOperations(memMapFs)
	operations := graphconfig.NewOperations(directoryOperations)
	modelPaths := operations.GetModelPaths(graphConfig)
	intersectSize := modelPaths.Intersect(expectedModelPaths).Size()
	// Check if the expected model paths are present
	if !(intersectSize == expectedModelPaths.Size() && intersectSize == modelPaths.Size()) {
		assert.Fail(t, fmt.Sprintf("some model paths which are expected not found in %s\n", modelPaths))
	}
}

func TestGetGraphConfigFromPath(t *testing.T) {
	memMapFs := afero.NewMemMapFs()
	fileContent := `
perception:
  detector_tracker:
    model_path: "/path/to/detector_tracker/model"
`
	testFilePath := "/config/cha.yaml"
	err := afero.WriteFile(memMapFs, testFilePath, []byte(fileContent), 0644)
	assert.NoError(t, err)

	directoryOperations := filesystem.NewDirectoryOperations(memMapFs)
	operations := graphconfig.NewOperations(directoryOperations)

	result, err := operations.LoadGraphConfigFromPath(testFilePath)
	assert.NoError(t, err)

	resultPath := result.Perception.DetectorTracker.GetModelPath()
	expectedPath := "/path/to/detector_tracker/model"
	assert.Equal(t, resultPath, expectedPath)
}

func TestGetGraphConfigFromDirectory(t *testing.T) {
	memMapFs := afero.NewMemMapFs()
	fileContent1 := `
perception:
  detector_tracker:
    model_path: "/path/to/detector_tracker/model1"
`
	fileContent2 := `
perception:
  detector_tracker:
    model_path: "/path/to/detector_tracker/model2"
`
	testDirectoryPath1 := "/configs1"
	testDirectoryPath2 := "/configs2"
	testFilePath1 := testDirectoryPath1 + "/cha.yaml"
	testFilePath11 := testDirectoryPath1 + "/cha1.yaml"
	testFilePath2 := testDirectoryPath2 + "/cha.yaml"

	err := memMapFs.MkdirAll(testDirectoryPath1, 0755)
	assert.NoError(t, err)
	err = memMapFs.MkdirAll(testDirectoryPath2, 0755)
	assert.NoError(t, err)

	err = afero.WriteFile(memMapFs, testFilePath1, []byte(fileContent1), 0644)
	assert.NoError(t, err)
	err = afero.WriteFile(memMapFs, testFilePath11, []byte(fileContent1), 0644)
	assert.NoError(t, err)

	err = afero.WriteFile(memMapFs, testFilePath2, []byte(fileContent2), 0644)
	assert.NoError(t, err)

	directoryOperations := filesystem.NewDirectoryOperations(memMapFs)
	operations := graphconfig.NewOperations(directoryOperations)

	result, err := operations.GetGraphConfigFromDirectory("/")
	assert.NoError(t, err)
	assert.Equal(t, 2, len(result))
}

func TestGetGraphConfigFileName(t *testing.T) {
	memMapFs := afero.NewMemMapFs()
	directoryOperations := filesystem.NewDirectoryOperations(memMapFs)
	operations := graphconfig.NewOperations(directoryOperations)

	expected := "cha.yaml"
	result := operations.GetGraphConfigFileName()

	if result != expected {
		t.Errorf("expected %s, but got %s", expected, result)
	}
}
