package main

import (
	"context"
	"fmt"
	"io"
	"net/url"
	"path/filepath"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

const (
	voxelTritonRepoMetaKey = "x-amz-meta-voxel-triton-repo"
)

var _ RepoFS = (*S3RepoFS)(nil)

// S3RepoFS provides apis for writing triton model repo files to S3
type S3RepoFS struct {
	ctx    context.Context
	client *s3.Client
	bucket string
	prefix string
}

// NewS3RepoFS constructs a repo filesystem handle that writes to S3
func NewS3RepoFS(ctx context.Context, repoPath string) (*S3RepoFS, error) {
	bucket, prefix, err := parseS3URL(repoPath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse repo url: %w", err)
	}

	awsConfig, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to load aws config: %w", err)
	}

	return &S3RepoFS{
		ctx:    ctx,
		bucket: bucket,
		prefix: prefix,
		client: s3.NewFromConfig(awsConfig),
	}, nil
}

func parseS3URL(rawuri string) (bucket, key string, err error) {
	s3path, err := url.Parse(rawuri)
	if err != nil {
		return "", "", fmt.Errorf("failed to parse s3 url: %w", err)
	}

	key = strings.TrimPrefix(s3path.Path, "/")
	if len(key) == 0 {
		return "", "", fmt.Errorf("invalid empty s3 key specified: %v", rawuri)
	}

	if len(s3path.Host) == 0 {
		return "", "", fmt.Errorf("invalid s3 bucket specified: %v", rawuri)
	}

	return s3path.Host, key, nil
}

func (f *S3RepoFS) path(key string) string {
	return filepath.Join(f.prefix, key)
}

func (f *S3RepoFS) meta() map[string]string {
	return map[string]string{
		voxelTritonRepoMetaKey: f.prefix,
	}
}

func (f *S3RepoFS) metavalid(meta map[string]string) bool {
	return meta[voxelTritonRepoMetaKey] == f.prefix
}

// WriteFile writes a model repo file to s3
func (f *S3RepoFS) WriteFile(key string, body io.Reader) error {
	_, err := f.client.PutObject(f.ctx, &s3.PutObjectInput{
		Bucket:   aws.String(f.bucket),
		Key:      aws.String(f.path(key)),
		Body:     body,
		Metadata: f.meta(),
	})
	if err != nil {
		return fmt.Errorf("s3 put object error: %w", err)
	}

	return nil
}

// ReadFile reads all file bytes for the passed in file from S3
func (f *S3RepoFS) ReadFile(key string) ([]byte, error) {
	resp, err := f.client.GetObject(f.ctx, &s3.GetObjectInput{
		Bucket: aws.String(f.bucket),
		Key:    aws.String(f.path(key)),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to read file %q from model repo: %w", key, err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if !f.metavalid(resp.Metadata) {
		return nil, fmt.Errorf("got invalid repo metadata for file %q: %w", key, err)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read file %q from model repo: %w", key, err)
	}

	return data, nil
}

// ListDirectory lists all files in the model repo at the specified prefix
func (f *S3RepoFS) ListDirectory(prefix string) ([]string, error) {
	files := []string{}

	listPrefix := filepath.Join(f.prefix, prefix) + "/"
	var continuationToken *string
	for {
		resp, err := f.client.ListObjectsV2(f.ctx, &s3.ListObjectsV2Input{
			Bucket:            aws.String(f.bucket),
			Prefix:            aws.String(listPrefix),
			ContinuationToken: continuationToken,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to list objects for repo: %w", err)
		}

		for _, obj := range resp.Contents {
			relname := strings.TrimPrefix(aws.ToString(obj.Key), f.prefix)
			relname = strings.TrimPrefix(relname, "/")
			files = append(files, relname)
		}

		// just some bounds checking so this can't consume an
		// unbounded amount of memory. typical model repos have 3-4
		// files per model so this would set an upper bound of ~25k models
		if len(files) > 100000 {
			return nil, fmt.Errorf("unsupported model repo has >100,000 files")
		}

		if resp.ContinuationToken == nil {
			break
		}

		continuationToken = resp.ContinuationToken
	}
	return files, nil
}

// RemoveAll removes all files from the specified prefix from S3. This call will
// fail if the object metadata for any objects do not appear to have come from this tool.
// Files manually uploaded to a model repo will have to be manually removed
func (f *S3RepoFS) RemoveAll(prefix string) error {
	files, err := f.ListDirectory(prefix)
	if err != nil {
		return fmt.Errorf("failed to list files for prefix %q: %w", prefix, err)
	}

	for _, filename := range files {
		fullpath := f.path(filename)

		headResp, err := f.client.HeadObject(f.ctx, &s3.HeadObjectInput{
			Bucket: aws.String(f.bucket),
			Key:    aws.String(fullpath),
		})
		if err != nil {
			return fmt.Errorf("HeadObject failed for %q: %w", filename, err)
		}

		if !f.metavalid(headResp.Metadata) {
			return fmt.Errorf("invalid object metadata for %q", filename)
		}

		_, err = f.client.DeleteObject(f.ctx, &s3.DeleteObjectInput{
			Bucket: aws.String(f.bucket),
			Key:    aws.String(fullpath),
		})
		if err != nil {
			return fmt.Errorf("failed to delete object %q: %w", fullpath, err)
		}
	}

	return nil
}
