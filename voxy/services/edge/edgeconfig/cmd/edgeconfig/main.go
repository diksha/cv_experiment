package main

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/secretsmanager"
	"github.com/aws/aws-sdk-go-v2/service/secretsmanager/types"
	"github.com/cristalhq/aconfig"
	"github.com/cristalhq/aconfig/aconfigyaml"
	"github.com/davecgh/go-spew/spew"

	"github.com/voxel-ai/voxel/lib/infra/edge/edgeconfig"
)

const (
	edgeAllowedUUIDPrimaryTag = "edge:allowed-uuid:primary"
	edgeConfigFilename        = "edgeconfig.yaml"
)

// Config holds the edgeconfig service configuration
type Config struct {
	LocalEdgeConfigPath string

	AWS struct {
		Region string
	}
	IOT struct {
		ThingName string
	}
}

func writeTempFile(dir, pattern string, data []byte) (filename string, err error) {
	outf, err := os.CreateTemp(dir, pattern)
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	defer func() {
		if err != nil {
			_ = outf.Close()
			_ = os.Remove(outf.Name())
		}
	}()

	n, err := outf.Write(data)
	if err != nil {
		return "", fmt.Errorf("failed to write temp file data: %w", err)
	}
	if n != len(data) {
		return "", fmt.Errorf("failed to write all bytes to temp file (%d < %d)", n, len(data))
	}

	return outf.Name(), outf.Close()
}

// atomicWriteFile will make sure that a file is written atomically and no consumer can read a partially written file
// this is important for configs because it is theoretically possible for a Write call to result in multile write system calls
// and so a config file read at just the wrong time will read a partially written configuration file.
func atomicWriteFile(filename string, data []byte, mode os.FileMode) error {
	absFilename, err := filepath.Abs(filename)
	if err != nil {
		return fmt.Errorf("invalid filepath %q: %w", filename, err)
	}

	filedir := filepath.Dir(absFilename)
	filebase := filepath.Base(absFilename)

	filetmp, err := writeTempFile(filedir, filebase, data)
	if err != nil {
		return fmt.Errorf("failed to write temporary file for %q: %w", filename, err)
	}

	err = os.Rename(filetmp, filename)
	if err != nil {
		// clean up our tmpfile if the rename fails
		_ = os.Remove(filetmp)
		return fmt.Errorf("failed to rename temporary file to %q -> %q: %w", filetmp, filename, err)
	}

	err = os.Chmod(filename, mode)
	if err != nil {
		return fmt.Errorf("failed to set file permissions: %w", err)
	}
	return nil
}

func fetchLocalConfig(edgeConfigPath string) (string, error) {
	data, err := ioutil.ReadFile(edgeConfigPath)
	if err != nil {
		return "", fmt.Errorf("failed to read local edge config file %q: %w", edgeConfigPath, err)
	}
	return string(data), nil
}

func fetchRemoteConfig(ctx context.Context, secretARN string, awsConfig aws.Config) (string, error) {
	data, err := fetchEdgeConfigSecret(ctx, secretARN, awsConfig)
	if err != nil {
		return "", fmt.Errorf("failed to pull remote edgeconfig.yaml: %w", err)
	}

	if _, err := edgeconfig.ParseYAML(data); err != nil {
		return "", fmt.Errorf("invalid edgeconfig.yaml found: %w", err)
	}

	return data, nil
}

func writeLocalConfig(edgeConfigPath string, data string) error {
	if err := atomicWriteFile(edgeConfigPath, []byte(data), 0644); err != nil {
		return fmt.Errorf("failed to write local edge config file %q: %w", edgeConfigPath, err)
	}
	return nil
}

// updateLocalConfig will update the local config file if necessary, returning
// true if it was updated and false in all other cases
func updateLocalConfig(ctx context.Context, secretARN, edgeConfigPath string, awsConfig aws.Config) (bool, error) {
	localConfig, err := fetchLocalConfig(edgeConfigPath)
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return false, fmt.Errorf("failed to fetch local config: %w", err)
	}

	remoteConfig, err := fetchRemoteConfig(ctx, secretARN, awsConfig)
	if err != nil {
		return false, fmt.Errorf("failed to fetch remote config: %w", err)
	}

	if localConfig == remoteConfig {
		return false, nil
	}

	if err := writeLocalConfig(edgeConfigPath, remoteConfig); err != nil {
		return false, fmt.Errorf("failed to write local config: %w", err)
	}

	return true, nil
}

func findEdgeConfigSecretARN(ctx context.Context, thingName string, awsConfig aws.Config) (string, error) {
	sm := secretsmanager.NewFromConfig(awsConfig)
	resp, err := sm.ListSecrets(ctx, &secretsmanager.ListSecretsInput{
		Filters: []types.Filter{
			{Key: "tag-value", Values: []string{thingName}},
		},
	})
	if err != nil {
		return "", fmt.Errorf("error fetching config from secrets manager: %w", err)
	}

	for _, entry := range resp.SecretList {
		for _, tag := range entry.Tags {
			key := aws.ToString(tag.Key)
			value := aws.ToString(tag.Value)
			name := aws.ToString(entry.Name)

			if key == edgeAllowedUUIDPrimaryTag && value == thingName && strings.HasSuffix(name, edgeConfigFilename) {
				return aws.ToString(entry.ARN), nil
			}
		}
	}

	return "", fmt.Errorf("no valid secrets found for %q", thingName)
}

func fetchEdgeConfigSecret(ctx context.Context, secretARN string, awsConfig aws.Config) (string, error) {
	sm := secretsmanager.NewFromConfig(awsConfig)
	res, err := sm.GetSecretValue(ctx, &secretsmanager.GetSecretValueInput{
		SecretId: aws.String(secretARN),
	})
	if err != nil {
		return "", fmt.Errorf("failed to get secret value for %q: %w", secretARN, err)
	}

	return aws.ToString(res.SecretString), nil
}

func main() {
	log.SetFlags(0)

	// This timer is used to ensure that we have waited the minimum amount of time between restarts
	// This number has to be larger than the `errorResetTime` property in the voxel.edge.EdgeConfig component
	// recipe under the lifecycle, which is currently set to 25 (which means 25s). Since this value is longer,
	// the "error restart" counter in greengrass will not be incremented as long as we wait longer than this value.
	minRestartTimer := time.After(30 * time.Second)

	var cfg Config
	loader := aconfig.LoaderFor(&cfg, aconfig.Config{
		FileDecoders: map[string]aconfig.FileDecoder{
			".yaml": aconfigyaml.New(),
		},
	})
	if err := loader.Load(); err != nil {
		log.Fatalf("failed to load app config: %v", err)
	}

	log.Printf("Starting up with config...")
	spew.Dump(cfg)

	awsConfig, err := config.LoadDefaultConfig(context.Background())
	if err != nil {
		log.Fatalf("failed to load aws config: %v", err)
	}

	secretARN, err := findEdgeConfigSecretARN(context.Background(), cfg.IOT.ThingName, awsConfig)
	if err != nil {
		log.Fatalf("failed to find edgeconfig.yaml for thingName=%s: %v", cfg.IOT.ThingName, err)
	}

	log.Printf("found edge config secret arn %q", secretARN)

	for {
		log.Printf("checking for secret updates...")
		updated, err := updateLocalConfig(context.Background(), secretARN, cfg.LocalEdgeConfigPath, awsConfig)
		if err != nil {
			log.Fatalf("edge config update failed: %v", err)
		}

		if updated {
			log.Printf("secret was updated, waiting for minimum restart time and restarting")
			// make sure we have waited the minimum restart time.
			<-minRestartTimer
			os.Exit(1)
		}

		log.Printf("secret was already up to date, waiting 5 minutes to check again")
		// wait 5 minutes between config polls since this API charges based
		time.Sleep(5 * time.Minute)
	}
}
