package edgeconfig

import (
	"encoding/json"
	"fmt"

	"google.golang.org/protobuf/encoding/protojson"

	edgeconfigpb "github.com/voxel-ai/voxel/protos/edge/edgeconfig/v1"

	"sigs.k8s.io/yaml"
)

// ParseYAML parses an edgeconfig.yaml file by first converting it to JSON
// We default to non-strict parsing for best compatibility as we add configuration data
func ParseYAML(data string) (*edgeconfigpb.EdgeConfig, error) {
	var yamlCfg interface{}
	if err := yaml.Unmarshal([]byte(data), &yamlCfg); err != nil {
		return nil, fmt.Errorf("failed to parse edgeconfig yaml: %v", err)
	}

	rawjson, err := json.Marshal(yamlCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to convert edgeconfig yaml to json: %v", err)
	}

	cfg := &edgeconfigpb.EdgeConfig{}
	if err := (protojson.UnmarshalOptions{DiscardUnknown: true}).Unmarshal(rawjson, cfg); err != nil {
		return nil, err
	}

	if cfg.Version > 1 {
		return nil, fmt.Errorf("invalid edge config version %d, must be <= 1", cfg.Version)
	}

	return cfg, nil
}
