package filesystem_test

import (
	"errors"
	"fmt"
	"github.com/stretchr/testify/assert"
	"github.com/voxel-ai/voxel/lib/utils/go/filesystem"
	"github.com/voxel-ai/voxel/lib/utils/go/filesystem/filesystemfakes"
	"google.golang.org/protobuf/proto"
	"sigs.k8s.io/yaml"
	"strings"
	"testing"

	cirruspb "github.com/voxel-ai/voxel/protos/perception/cirrus/v1"
)

func setup(fakeDirectory *filesystemfakes.FakeDirectoryOperationsInterface) {
	fakeDirectory.IsDirectoryStub = func(string) (bool, error) {
		return true, nil
	}
	fakeDirectory.IsFileStub = func(string) (bool, error) {
		return true, nil
	}

	fakeDirectory.GetFilePathsStub = func(path string, ext string) ([]string, error) {
		if ext != ".yaml" {
			return nil, errors.New("invalid arguments")
		}
		switch path {
		case "/path/to/yaml/files/services":
			return []string{"/path/to/yaml/files/services/service1.yaml", "/path/to/yaml/files/services/service2.yaml"}, nil
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
			data, _ := yaml.JSONToYAML([]byte("{\n  \"models\": {\n    \"artifact_model_paths\": [\n      \"path1\",\n      \"path2\"\n    ]\n  }\n}"))
			return data, nil
		default:
			return nil, fmt.Errorf("invalid file path: %s", path)
		}
	}
}

func TestGetYamlObjectFromFilePath(t *testing.T) {
	testCases := []struct {
		name          string
		path          string
		readFileErr   error
		expectedErr   error
		expectedProto proto.Message
	}{
		{
			name:          "Success",
			path:          "/path/to/yaml/files/services/service1.yaml",
			expectedProto: &cirruspb.TritonModel{},
		},
		{
			name:          "Failed to read file",
			path:          "/path/to/yaml/files/services/service3.yaml",
			readFileErr:   errors.New("failed to read the file: invalid file path: /path/to/yaml/files/services/service3.yaml"),
			expectedErr:   errors.New("failed to read the file: invalid file path: /path/to/yaml/files/services/service3.yaml"),
			expectedProto: &cirruspb.TritonModel{},
		},
		{
			name:          "Failed to parse config protobuf",
			path:          "/path/to/yaml/files/services/service2.yaml",
			expectedErr:   errors.New("unknown field \"models\""),
			expectedProto: &cirruspb.TritonModel{},
		},
		{
			name:        "Failed to parse config protobuf",
			path:        "/path/to/yaml/files/services/service2.yaml",
			expectedErr: errors.New("proto message set to nil"),
		},
	}

	for _, testcase := range testCases {
		t.Run(testcase.name, func(t *testing.T) {
			fakeDir := new(filesystemfakes.FakeDirectoryOperationsInterface)
			setup(fakeDir)
			convertor := filesystem.NewConvertor(fakeDir)
			err := convertor.GetYamlObjectFromFilePath(testcase.path, testcase.expectedProto)

			if testcase.expectedErr == nil {
				assert.NoError(t, err)
			} else {
				assert.True(t, strings.Contains(err.Error(), testcase.expectedErr.Error()))
			}
		})
	}
}
