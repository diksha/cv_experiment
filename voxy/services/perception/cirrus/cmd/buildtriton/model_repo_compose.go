//Copyright 2023 Voxel Labs, Inc.
//All rights reserved.
//
//This document may not be reproduced, republished, distributed, transmitted,
//displayed, broadcast or otherwise exploited in any manner without the express
//prior written permission of Voxel Labs, Inc. The receipt or possession of this
//document does not convey any rights to reproduce, disclose, or distribute its
//contents, or to manufacture, use, or sell anything that it may describe, in
//whole or in part.

// Package buildtriton is a tool to generate Triton model repository configuration files.
package buildtriton

import (
	"github.com/hashicorp/go-set"
	"github.com/voxel-ai/voxel/lib/graphconfig"
	"github.com/voxel-ai/voxel/lib/utils/go/filesystem"
	cirruspb "github.com/voxel-ai/voxel/protos/perception/cirrus/v1"
	graphconfigpb "github.com/voxel-ai/voxel/protos/perception/graph_config/v1"
)

// ModelRepoInterface interface is used to construct the repo models
//
//go:generate go run github.com/maxbrunsfeld/counterfeiter/v6 . ModelRepoInterface
type ModelRepoInterface interface {
	GetModelsFromTritonConfig(modelMap map[string]*cirruspb.Model, graphConfigMap map[string]*graphconfigpb.GraphConfig) ([]*cirruspb.Model, error)
	GetEnsemblesFromTritonConfig(ensembleMap map[string]*cirruspb.Ensemble, graphConfigMap map[string]*graphconfigpb.GraphConfig) ([]*cirruspb.Ensemble, error)
}

// ModelRepo struct is used to construct the repo models
type ModelRepo struct {
	modelRepoInterface  ModelRepoInterface
	operationsInterface graphconfig.OperationsInterface
}

// NewModelRepo returns a new ModelRepoInterface interface
func NewModelRepo(directory filesystem.DirectoryOperationsInterface) *ModelRepo {
	return &ModelRepo{modelRepoInterface: &ModelRepo{}, operationsInterface: graphconfig.NewOperations(directory)}
}

// GetModelsFromTritonConfig returns the models from the models defined in the model config directory
func (mtc *ModelRepo) GetModelsFromTritonConfig(modelMap map[string]*cirruspb.Model, graphConfigMap map[string]*graphconfigpb.GraphConfig) ([]*cirruspb.Model, error) {
	var graphConfigArtifactPath *set.Set[string]
	for _, value := range graphConfigMap {
		graphConfigArtifactPath = mtc.operationsInterface.GetModelPaths(value)
	}
	var modelList []*cirruspb.Model
	for _, value := range modelMap {
		result := mtc.GetModelsFromArtifactPathList(value, graphConfigArtifactPath)
		if result == nil {
			continue
		}
		modelList = append(modelList, result)
	}
	return modelList, nil
}

// GetEnsemblesFromTritonConfig returns the models from the models for the defined graph config master yaml
func (mtc *ModelRepo) GetEnsemblesFromTritonConfig(ensembleMap map[string]*cirruspb.Ensemble, graphConfigMap map[string]*graphconfigpb.GraphConfig) ([]*cirruspb.Ensemble, error) {
	var graphConfigArtifactPath *set.Set[string]
	for _, value := range graphConfigMap {
		graphConfigArtifactPath = mtc.operationsInterface.GetModelPaths(value)
	}
	var ensembleList []*cirruspb.Ensemble
	for _, value := range ensembleMap {
		result := mtc.GetEnsemblesFromArtifactPathList(value, graphConfigArtifactPath)
		if result == nil {
			continue
		}
		ensembleList = append(ensembleList, result)
	}
	return ensembleList, nil
}

// GetModelsFromArtifactPathList returns the models from the models defined in the model config directory
func (mtc *ModelRepo) GetModelsFromArtifactPathList(model *cirruspb.Model, graphConfigArtifactPath *set.Set[string]) *cirruspb.Model {
	var outputArtifactList []string

	for _, value := range model.GetArtifactModelPaths() {
		if graphConfigArtifactPath.Contains(value) {
			outputArtifactList = append(outputArtifactList, value)
		}
	}

	if len(outputArtifactList) > 0 {
		model.ArtifactModelPaths = outputArtifactList
		return model
	}
	return nil
}

// GetEnsemblesFromArtifactPathList returns the models from the models defined in the graph config master yaml
// TODO : Need to figure out what needs to be done here
func (mtc *ModelRepo) GetEnsemblesFromArtifactPathList(ensemble *cirruspb.Ensemble, graphConfigArtifactPath *set.Set[string]) *cirruspb.Ensemble {
	var outputArtifactList []string

	for _, value := range ensemble.GetArtifactModelPaths() {
		if graphConfigArtifactPath.Contains(value) {
			outputArtifactList = append(outputArtifactList, value)
		}
	}

	if len(outputArtifactList) > 0 {
		ensemble.ArtifactModelPaths = outputArtifactList
		return ensemble
	}
	return nil
}
