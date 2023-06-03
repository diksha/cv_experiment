package main

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
)

var _ RepoFS = (*OSRepoFS)(nil)

// OSRepoFS provides an interface for writing model repo files to a filesystem
type OSRepoFS struct {
	prefix string
}

func newOSRepoFS(repoPath string) (*OSRepoFS, error) {
	if err := os.MkdirAll(repoPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create repo directory: %w", err)
	}

	return &OSRepoFS{prefix: repoPath}, nil
}

// WriteFile writes model repo files to the filesystem
func (f *OSRepoFS) WriteFile(key string, body io.Reader) error {
	filename := path.Join(f.prefix, key)
	dirname, _ := path.Split(filename)

	if err := os.MkdirAll(dirname, 0755); err != nil {
		return fmt.Errorf("failed to make model directory: %w", err)
	}

	outf, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to write file %v: %w", filename, err)
	}
	defer func() {
		_ = outf.Close()
	}()

	if _, err := io.Copy(outf, body); err != nil {
		return fmt.Errorf("failed to write file %w", err)
	}

	if err := outf.Close(); err != nil {
		return fmt.Errorf("failed to write file %w", err)
	}

	return nil
}

// RemoveAll removes all files from the repo prefix with the provided prefix
func (f *OSRepoFS) RemoveAll(name string) error {
	if name == "" {
		return fmt.Errorf("invalid empty name passed to RemoveAll")
	}

	removePath := filepath.Join(f.prefix, name)

	if err := os.RemoveAll(removePath); err != nil {
		return fmt.Errorf("failed to remove path %q", name)
	}
	return nil
}

// ListDirectory lists the model repo files located at the passed in prefix
func (f *OSRepoFS) ListDirectory(prefix string) ([]string, error) {
	files := []string{}

	err := filepath.Walk(filepath.Join(f.prefix, prefix), func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			return fmt.Errorf("failed to list files: %w", err)
		}

		if !info.IsDir() {
			relname := strings.TrimPrefix(path, f.prefix)
			relname = strings.TrimPrefix(relname, "/")
			files = append(files, relname)
		}

		return nil
	})
	if errors.Is(err, os.ErrNotExist) {
		return files, nil
	} else if err != nil {
		return nil, fmt.Errorf("failed to list files: %w", err)
	}

	return files, nil
}

// ReadFile reads all bytes for the passed in file from disk
func (f *OSRepoFS) ReadFile(name string) ([]byte, error) {
	data, err := os.ReadFile(filepath.Join(f.prefix, name))
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}
	return data, nil
}
