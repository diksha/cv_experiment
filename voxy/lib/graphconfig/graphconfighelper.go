// Package graphconfig contains utility functions for graph config operations.
package graphconfig

import (
	"fmt"
	"strings"

	"google.golang.org/protobuf/reflect/protoreflect"

	"github.com/hashicorp/go-set"

	"github.com/voxel-ai/voxel/lib/utils/go/filesystem"
	graphconfigpb "github.com/voxel-ai/voxel/protos/perception/graph_config/v1"
)

const modelPath protoreflect.Name = "model_path"

// OperationsInterface is used to provide operations on graph config
type OperationsInterface interface {
	GetModelPaths(graphConfig *graphconfigpb.GraphConfig) *set.Set[string]
	LoadGraphConfigFromPath(filePath string) (*graphconfigpb.GraphConfig, error)
	GetGraphConfigFromDirectory(dirPath string) (map[string]*graphconfigpb.GraphConfig, error)
	GetGraphConfigFileName() string
}

// Operations struct is used to provide operations on graph config
type Operations struct {
	operations          OperationsInterface
	convertor           filesystem.ConvertorInterface
	directoryOperations filesystem.DirectoryOperationsInterface
}

// NewOperations returns a new OperationsInterface
func NewOperations(directory filesystem.DirectoryOperationsInterface) *Operations {
	return &Operations{operations: &Operations{}, directoryOperations: directory, convertor: filesystem.NewConvertor(directory)}
}

// Recursively find and print the "model_path" field values
func (o *Operations) findModelPath(field protoreflect.FieldDescriptor, value protoreflect.Value, modelPathsInGraphConfig *set.Set[string]) *set.Set[string] {
	switch field.Kind() {
	case protoreflect.MessageKind:
		message := value.Message()
		message.Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
			o.findModelPath(fd, v, modelPathsInGraphConfig)
			return true
		})
	case protoreflect.StringKind:
		if field.Name() == modelPath {
			modelPathsInGraphConfig.Insert(value.String())
		}
	}
	return modelPathsInGraphConfig
}

// GetModelPaths recursively processes the protobuf message
func (o *Operations) GetModelPaths(graphConfig *graphconfigpb.GraphConfig) *set.Set[string] {
	modelPathsInGraphConfig := set.New[string](0)
	msg := graphConfig.ProtoReflect()
	// Iterate through the fields
	msg.Range(func(field protoreflect.FieldDescriptor, value protoreflect.Value) bool {
		modelPathsInGraphConfig = o.findModelPath(field, value, modelPathsInGraphConfig)
		return true
	})
	return modelPathsInGraphConfig
}

// LoadGraphConfigFromPath returns the graph config from the path
func (o *Operations) LoadGraphConfigFromPath(filePath string) (*graphconfigpb.GraphConfig, error) {
	if !strings.Contains(filePath, o.GetGraphConfigFileName()) {
		return nil, fmt.Errorf("graph config file name should be %s", o.GetGraphConfigFileName())
	}
	cfg := &graphconfigpb.GraphConfig{}
	if err := o.convertor.GetYamlObjectFromFilePath(filePath, cfg); err != nil {
		return nil, fmt.Errorf("failed to convert the yaml file to obj: %w", err)
	}
	return cfg, nil
}

// GetGraphConfigFromDirectory returns the graph config from the directory
func (o *Operations) GetGraphConfigFromDirectory(dirPath string) (map[string]*graphconfigpb.GraphConfig, error) {
	graphConfig := make(map[string]*graphconfigpb.GraphConfig)
	paths, err := o.directoryOperations.GetFilePaths(dirPath, ".yaml")
	if err != nil {
		return nil, fmt.Errorf("error walking through directory: %w", err)
	}
	for _, path := range paths {
		if !strings.Contains(path, o.GetGraphConfigFileName()) {
			continue
		}
		cfg, err := o.LoadGraphConfigFromPath(path)
		if err != nil {
			return nil, err
		}
		graphConfig[path] = cfg
	}
	return graphConfig, nil
}

// GetGraphConfigFileName returns the graph config file name
func (o *Operations) GetGraphConfigFileName() string {
	return "cha.yaml"
}
