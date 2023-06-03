// Package allows to fetch a github token from a github app and print it to stdout.
package main

import (
	"fmt"
	"log"

	"github.com/cristalhq/aconfig"

	"github.com/voxel-ai/voxel/services/platform/github/lib/token"
)

// Config holds the configuration values for the GitHub App.
type Config struct {
	GithubAppID          int64  `required:"true"`
	GithubInstallationID string `required:"true"`
	GithubPrivateKey     string `required:"true"`
}

func main() {
	var cfg Config
	loader := aconfig.LoaderFor(&cfg, aconfig.Config{
		SkipDefaults: true,
		SkipFiles:    false,
		SkipEnv:      false,
		Files:        []string{"config.yml", "config.json", "config.toml"},
		EnvPrefix:    "",
		FlagPrefix:   "",
	})
	if err := loader.Load(); err != nil {
		log.Fatalf("Failed to load configuration: %v\n", err)
	}

	accessToken, err := token.CreateGitHubToken(cfg.GithubAppID, cfg.GithubInstallationID, cfg.GithubPrivateKey)
	if err != nil {
		log.Fatalf("Failed to create GitHub token: %v\n", err)
	}
	fmt.Println(accessToken)
}
