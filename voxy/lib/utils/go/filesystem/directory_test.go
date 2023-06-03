package filesystem_test

import (
	"errors"
	"fmt"
	"github.com/voxel-ai/voxel/lib/utils/go/filesystem"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/spf13/afero"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewDirectory(t *testing.T) {
	fs := afero.NewMemMapFs()

	dir := filesystem.NewDirectoryOperations(fs)

	assert.NotNil(t, dir)
}

func TestReadFilesOnlyYamlFiles(t *testing.T) {
	path := "/test"
	fs := afero.NewMemMapFs()
	require.NoError(t, fs.MkdirAll(path+"/dir1", 0755))
	require.NoError(t, afero.WriteFile(fs, path+"/dir1/file1.yaml", []byte("file1"), 0644))
	require.NoError(t, afero.WriteFile(fs, path+"/dir1/file2.yaml", []byte("file2"), 0644))
	require.NoError(t, fs.MkdirAll(path+"/dir2", 0755))
	require.NoError(t, afero.WriteFile(fs, path+"/dir2/file3.yaml", []byte("file3"), 0644))
	require.NoError(t, afero.WriteFile(fs, path+"/dir2/file4.txt", []byte("file4"), 0644))

	dir := filesystem.NewDirectoryOperations(fs)

	fileList, err := dir.GetFilePaths(path, ".yaml")
	require.NoError(t, err)

	assert.Len(t, fileList, 3)
	assert.Contains(t, fileList, "/test/dir1/file1.yaml")
	assert.Contains(t, fileList, "/test/dir1/file2.yaml")
	assert.Contains(t, fileList, "/test/dir2/file3.yaml")
	assert.NotContains(t, fileList, "/test/dir2/file3.txt")
}

func TestReadFilesOnlyYamlFilesAccessingPath(t *testing.T) {
	fs := afero.NewMemMapFs()
	dir := filesystem.NewDirectoryOperations(fs)
	_, err := dir.GetFilePaths("/nonexistent", "yaml")

	assert.Error(t, err)
	assert.True(t, errors.Is(err, os.ErrNotExist))
}

type testErrorFs struct {
	afero.Fs
}

func (t *testErrorFs) LstatIfPossible(path string) (os.FileInfo, bool, error) {
	return nil, false, errors.New("test error walking directory")
}

func TestReadFilesOnlyYamlFilesErrorWalkingDirectory(t *testing.T) {
	path := "/test"
	fs := afero.NewMemMapFs()
	require.NoError(t, fs.MkdirAll(path+"/dir1", 0755))
	require.NoError(t, afero.WriteFile(fs, path+"/dir1/file1.yaml", []byte("file1"), 0644))
	require.NoError(t, afero.WriteFile(fs, path+"/dir1/file2.yaml", []byte("file2"), 0644))
	require.NoError(t, fs.MkdirAll(path+"/dir2", 0755))
	require.NoError(t, afero.WriteFile(fs, path+"/dir2/file3.txt", []byte("file3"), 0644))

	fsErr := &testErrorFs{Fs: fs} // Use the custom filesystem implementation
	dir := filesystem.NewDirectoryOperations(fsErr)

	_, err := dir.GetFilePaths(path, "yaml")

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "test error walking directory")
}

func TestGetAllYamlFilePathsInDirectory_MaxFilesCount(t *testing.T) {
	// Create a temporary in-memory filesystem with afero
	appFS := afero.NewMemMapFs()

	// Create a directory and multiple YAML files
	testDir := "/test"
	err := appFS.Mkdir(testDir, 0755)
	if err != nil {
		t.Fatalf("Failed to create test directory: %v", err)
	}

	// Add 2050 YAML files to the test directory
	for i := 1; i <= 2050; i++ {
		filename := filepath.Join(testDir, "file"+strconv.Itoa(i)+".yaml")
		err := afero.WriteFile(appFS, filename, []byte("test"), 0644)
		if err != nil {
			t.Fatalf("Failed to create test YAML file: %v", err)
		}
	}

	d := filesystem.DirectoryOperations{FileSystem: appFS, MaxFilesCount: 2048}

	// Attempt to read all YAML files in the directory
	_, err = d.GetFilePaths(testDir, ".yaml")

	// Check for the expected error due to exceeding MaxFilesCount
	if err == nil || !strings.Contains(err.Error(), "max number of files count reached") {
		t.Fatalf("Expected max number of files count error, but got: %v", err)
	}
}

func errorComparator(expected, actual error) bool {
	return errors.Is(actual, expected) || fmt.Sprint(expected) == fmt.Sprint(actual)
}

func TestReadFile(t *testing.T) {
	tests := []struct {
		name            string
		setupFs         func(fs afero.Fs) error
		path            string
		expected        []byte
		expectedErr     error
		errorComparator func(expected, actual error) bool
	}{
		{
			name: "successful_read",
			setupFs: func(fs afero.Fs) error {
				if err := fs.MkdirAll("/dir", 0755); err != nil {
					return fmt.Errorf("failed to create directory: %w", err)
				}
				if err := afero.WriteFile(fs, "/dir/file.txt", []byte("hello, world"), 0644); err != nil {
					return fmt.Errorf("failed to write file: %w", err)
				}
				return nil
			},
			path:     "/dir/file.txt",
			expected: []byte("hello, world"),
		},
		{
			name: "file_not_found",
			setupFs: func(fs afero.Fs) error {
				if err := fs.MkdirAll("/dir", 0755); err != nil {
					return fmt.Errorf("failed to create directory: %w", err)
				}
				return nil
			},
			path:            "/dir/nonexistent.txt",
			expectedErr:     fmt.Errorf("failed to read the file: open %s: %w", "/dir/nonexistent.txt", afero.ErrFileNotFound),
			errorComparator: errorComparator,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fs := afero.NewMemMapFs()
			err := test.setupFs(fs)
			if err != nil {
				t.Fatalf("failed to set up file system: %v", err)
			}

			dir := filesystem.DirectoryOperations{FileSystem: fs}
			data, err := dir.ReadFile(test.path)

			if test.expectedErr != nil {
				if test.errorComparator != nil {
					assert.True(t, test.errorComparator(test.expectedErr, err))
				} else {
					assert.ErrorIs(t, err, test.expectedErr)
				}
			} else {
				assert.NoError(t, err)
				assert.Equal(t, test.expected, data)
			}
		})
	}
}

func TestIsFile(t *testing.T) {
	memMapFs := afero.NewMemMapFs()
	dir := filesystem.NewDirectoryOperations(memMapFs)

	// Create a file
	filePath := "path/to/file.txt"
	file, err := memMapFs.Create(filePath)
	if err != nil {
		t.Fatalf("error creating test file: %s", err)
	}
	err = file.Close()
	assert.NoError(t, err)

	// Check if the file exists and is a file
	isFile, err := dir.IsFile(filePath)
	if err != nil {
		t.Fatalf("error checking if file is a file: %s", err)
	}
	if !isFile {
		t.Errorf("expected file to be a file")
	}

	// Check if a directory exists and is not a file
	dirPath := "path/to/directory"
	err = memMapFs.Mkdir(dirPath, 0755)
	if err != nil {
		t.Fatalf("error creating test directory: %s", err)
	}
	isFile, err = dir.IsFile(dirPath)
	if err != nil {
		t.Fatalf("error checking if directory is a file: %s", err)
	}
	if isFile {
		t.Errorf("expected directory to not be a file")
	}
}

func TestIsDirectory(t *testing.T) {
	memMapFs := afero.NewMemMapFs()
	dir := filesystem.NewDirectoryOperations(memMapFs)

	// Create a directory
	dirPath := "path/to/directory"
	err := memMapFs.Mkdir(dirPath, 0755)
	if err != nil {
		t.Fatalf("error creating test directory: %s", err)
	}

	// Check if the directory exists and is a directory
	isDir, err := dir.IsDirectory(dirPath)
	if err != nil {
		t.Fatalf("error checking if directory is a directory: %s", err)
	}
	if !isDir {
		t.Errorf("expected directory to be a directory")
	}

	// Check if a file exists and is not a directory
	filePath := "path/to/file.txt"
	file, err := memMapFs.Create(filePath)
	if err != nil {
		t.Fatalf("error creating test file: %s", err)
	}
	err = file.Close()
	assert.NoError(t, err)
	isDir, err = dir.IsDirectory(filePath)
	if err != nil {
		t.Fatalf("error checking if file is a directory: %s", err)
	}
	if isDir {
		t.Errorf("expected file to not be a directory")
	}
}
