//Copyright 2023 Voxel Labs, Inc.
//All rights reserved.
//
//This document may not be reproduced, republished, distributed, transmitted,
//displayed, broadcast or otherwise exploited in any manner without the express
//prior written permission of Voxel Labs, Inc. The receipt or possession of this
//document does not convey any rights to reproduce, disclose, or distribute its
//contents, or to manufacture, use, or sell anything that it may describe, in
//whole or in part.

// Package filesystem contains utility functions for directory operations.
package filesystem

import (
	"fmt"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
	"sigs.k8s.io/yaml"
)

// ConvertorInterface interface is used to convert to protobuf objects
type ConvertorInterface interface {
	GetYamlObjectFromFilePath(path string, m proto.Message) error
}

// Convertor struct is used to convert to protobuf objects
type Convertor struct {
	convertor ConvertorInterface
	directory DirectoryOperationsInterface
}

// NewConvertor returns a new ConvertorInterface interface
func NewConvertor(directory DirectoryOperationsInterface) *Convertor {
	return &Convertor{
		convertor: &Convertor{},
		directory: directory,
	}
}

// GetYamlObjectFromFilePath returns the protobuf object from the yaml file
func (c *Convertor) GetYamlObjectFromFilePath(path string, message proto.Message) error {
	if message == nil {
		return fmt.Errorf("proto message set to nil")
	}
	data, err := c.directory.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read the file: %w", err)
	}

	json, err := yaml.YAMLToJSON(data)
	if err != nil {
		return fmt.Errorf("failed to marshal config JSON: %w", err)
	}

	if err := protojson.Unmarshal(json, message); err != nil {
		return fmt.Errorf("failed to parse config protobuf: %w", err)
	}
	return nil
}
