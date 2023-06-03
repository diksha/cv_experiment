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
	"os"
	"path/filepath"

	"github.com/spf13/afero"
)

// DirectoryOperationsInterface defines the methods that can be performed on a directory.
//
//go:generate go run github.com/maxbrunsfeld/counterfeiter/v6 . DirectoryOperationsInterface
type DirectoryOperationsInterface interface {
	GetFilePaths(path string, ext string) ([]string, error)
	ReadFile(path string) ([]byte, error)
	IsFile(path string) (bool, error)
	IsDirectory(path string) (bool, error)
}

// DirectoryOperations represents a directory in the filesystem.
type DirectoryOperations struct {
	FileSystem    afero.Fs
	MaxFilesCount int
}

// NewDirectoryOperations creates a new instance of DirectoryOperations with the given path and filesystem.
func NewDirectoryOperations(fs afero.Fs) *DirectoryOperations {
	return &DirectoryOperations{FileSystem: fs, MaxFilesCount: 2048}
}

// GetFilePaths reads all files from the directory and returns a list of file paths.
// It will log an error message if there is an issue accessing a specific path.
func (d *DirectoryOperations) GetFilePaths(path string, ext string) ([]string, error) {
	var fileList []string

	walkFunc := func(path string, info os.FileInfo, err error) error {
		if len(fileList) > d.MaxFilesCount {
			return fmt.Errorf("max number of files count reached, consider calling this function with lesser number of files")
		}
		if err != nil {
			return fmt.Errorf("error accessing path: %s, Error: %w", path, err)
		}
		if !info.IsDir() && filepath.Ext(path) == ext {
			absPath, err := filepath.Abs(path)

			if err != nil {
				return fmt.Errorf("error getting absolute path: %s, Error: %w", path, err)
			}
			fileList = append(fileList, absPath)
		}
		return nil
	}
	err := afero.Walk(d.FileSystem, path, walkFunc)
	if err != nil {
		return nil, fmt.Errorf("error walking through directory: %w", err)
	}

	return fileList, nil
}

// ReadFile reads the file from the given path and returns the file contents.
func (d *DirectoryOperations) ReadFile(path string) ([]byte, error) {
	data, err := afero.ReadFile(d.FileSystem, path)
	if err != nil {
		return nil, fmt.Errorf("failed to read the file: %w", err)
	}
	return data, nil
}

// IsFile returns true if the given path is a file.
func (d *DirectoryOperations) IsFile(path string) (bool, error) {
	info, err := d.FileSystem.Stat(path)
	if err != nil {
		return false, fmt.Errorf("failed to get state of the file: %w", err)
	}
	return !info.IsDir(), nil
}

// IsDirectory returns true if the given path is a directory.
func (d *DirectoryOperations) IsDirectory(path string) (bool, error) {
	info, err := d.FileSystem.Stat(path)
	if err != nil {
		return false, fmt.Errorf("failed to get state of the file: %w", err)
	}
	return info.IsDir(), nil
}
