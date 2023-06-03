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
	"fmt"
	graphconfigpb "github.com/voxel-ai/voxel/protos/perception/graph_config/v1"
	"strings"

	"github.com/voxel-ai/voxel/lib/graphconfig"
	"github.com/voxel-ai/voxel/lib/utils/go/filesystem"
	cirruspb "github.com/voxel-ai/voxel/protos/perception/cirrus/v1"
)

// TritonConfigInterface interface is used to construct the model yaml files
//
//go:generate go run github.com/maxbrunsfeld/counterfeiter/v6 . TritonConfigInterface
type TritonConfigInterface interface {
	GetModelsFromConfig(dirPath string) (map[string]*cirruspb.Model, map[string]*cirruspb.Ensemble, error)
	GetProductionGraphModelRepoFromConfig(dirPath string) (map[string]*cirruspb.ProductionGraphModelRepo, error)
	GetGraphConfigFromGraphModelRepo(graphModelRepoMap map[string]*cirruspb.ProductionGraphModelRepo) (map[string]*graphconfigpb.GraphConfig, error)
}

// TritonConfig struct is used to construct the model yaml files
type TritonConfig struct {
	tritonConfig      TritonConfigInterface
	directory         filesystem.DirectoryOperationsInterface
	convertor         filesystem.ConvertorInterface
	graphConfigHelper graphconfig.OperationsInterface
}

// NewTritonConfig returns a new TritonConfigInterface interface
func NewTritonConfig(directory filesystem.DirectoryOperationsInterface) *TritonConfig {
	return &TritonConfig{
		tritonConfig:      &TritonConfig{},
		directory:         directory,
		convertor:         filesystem.NewConvertor(directory),
		graphConfigHelper: graphconfig.NewOperations(directory),
	}
}

// GetModelsFromConfig returns the models and ensembles from the model config directory
func (tc *TritonConfig) GetModelsFromConfig(dirPath string) (map[string]*cirruspb.Model, map[string]*cirruspb.Ensemble, error) {
	models := make(map[string]*cirruspb.Model)
	ensembles := make(map[string]*cirruspb.Ensemble)
	paths, err := tc.directory.GetFilePaths(dirPath, ".yaml")
	if err != nil {
		return nil, nil, fmt.Errorf("error walking through directory: %w", err)
	}
	for _, path := range paths {
		cfg := &cirruspb.TritonModel{}
		if err := tc.convertor.GetYamlObjectFromFilePath(path, cfg); err != nil {
			return nil, nil, fmt.Errorf("failed to convert the yaml file to obj: %w", err)
		}
		key := tc.keyForTritonObjsMap(path)
		if cfg.GetModel() != nil {
			models[key] = cfg.GetModel()
		}
		if cfg.GetEnsemble() != nil {
			ensembles[key] = cfg.GetEnsemble()
		}
	}
	return models, ensembles, nil
}

// GetProductionGraphModelRepoFromConfig returns the production graph model repo from the production graph model repo config directory
func (tc *TritonConfig) GetProductionGraphModelRepoFromConfig(dirPath string) (map[string]*cirruspb.ProductionGraphModelRepo, error) {
	productionGraphModelConfig := make(map[string]*cirruspb.ProductionGraphModelRepo)
	paths, err := tc.directory.GetFilePaths(dirPath, ".yaml")
	if err != nil {
		return nil, fmt.Errorf("error walking through directory: %w", err)
	}
	for _, path := range paths {
		cfg := &cirruspb.ProductionGraphModelRepo{}
		if err := tc.convertor.GetYamlObjectFromFilePath(path, cfg); err != nil {
			return nil, fmt.Errorf("failed to convert the yaml file to obj: %w", err)
		}
		if len(cfg.GraphConfigPaths) == 0 || cfg.Name == "" {
			return nil, fmt.Errorf("either modelRepoName or GraphConfigPath is not set in the graph model repo: %s", tc.keyForTritonObjsMap(path))
		}
		productionGraphModelConfig[tc.keyForTritonObjsMap(path)] = cfg
	}
	return productionGraphModelConfig, nil
}

// GetGraphConfigFromGraphModelRepo returns the graph config from the graph model repo config directory
func (tc *TritonConfig) GetGraphConfigFromGraphModelRepo(graphModelRepoMap map[string]*cirruspb.ProductionGraphModelRepo) (map[string]*graphconfigpb.GraphConfig, error) {
	graphModelMap := make(map[string]*graphconfigpb.GraphConfig)
	var err error
	for _, graphModelRepo := range graphModelRepoMap {
		for _, path := range graphModelRepo.GetGraphConfigPaths() {
			graphModelMap, err = tc.processGraphConfigPath(path, graphModelMap)
			if err != nil {
				return nil, err
			}
		}
	}
	return graphModelMap, nil
}

func (tc *TritonConfig) processGraphConfigPath(path string, configMap map[string]*graphconfigpb.GraphConfig) (map[string]*graphconfigpb.GraphConfig, error) {
	isFile, err := tc.directory.IsFile(path)
	if err != nil {
		return nil, fmt.Errorf("error while check if the path is a file : %w", err)
	}

	if isFile {
		return tc.processGraphConfigFile(path, configMap)
	}
	return tc.processGraphConfigDirectory(path, configMap)

}

func (tc *TritonConfig) processGraphConfigFile(path string, configMap map[string]*graphconfigpb.GraphConfig) (map[string]*graphconfigpb.GraphConfig, error) {
	if !strings.Contains(path, tc.graphConfigHelper.GetGraphConfigFileName()) {
		return nil, fmt.Errorf("file is not a graph config : %s", path)
	}

	cfg, err := tc.graphConfigHelper.LoadGraphConfigFromPath(path)
	if err != nil {
		return nil, fmt.Errorf("failed to get graph config from file path : %w", err)
	}
	if configMap[path] != nil {
		return nil, fmt.Errorf("duplicate path found in config map: %s", path)
	}

	configMap[path] = cfg
	return configMap, nil
}

func (tc *TritonConfig) processGraphConfigDirectory(path string, configMap map[string]*graphconfigpb.GraphConfig) (map[string]*graphconfigpb.GraphConfig, error) {
	cfg, err := tc.graphConfigHelper.GetGraphConfigFromDirectory(path)
	if err != nil {
		return nil, fmt.Errorf("failed to get graph config from directory : %w", err)
	}

	for key, value := range cfg {
		if configMap[key] != nil {
			return nil, fmt.Errorf("duplicate path found in config map: %s", key)
		}
		configMap[key] = value
	}

	return configMap, nil
}

// Techdebt..!! TODO clean this.
func (tc *TritonConfig) keyForTritonObjsMap(path string) string {
	return "services" + strings.Split(path, "services")[len(strings.Split(path, "services"))-1]
}
