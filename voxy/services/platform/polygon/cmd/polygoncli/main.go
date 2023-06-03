// package main creates the connection to the server and passes it to cobra CLI
package main

import (
	"github.com/cristalhq/aconfig"

	"github.com/voxel-ai/voxel/services/platform/polygon/cmd/polygoncli/polygoncmd"

	"context"

	"time"

	"github.com/rs/zerolog/log"

	polygonserver "github.com/voxel-ai/voxel/services/platform/polygon/lib/server"
)

func main() {
	var cfg polygonserver.Config
	loader := aconfig.LoaderFor(&cfg, aconfig.Config{EnvPrefix: "POLYGON", SkipFlags: true})
	if err := loader.Load(); err != nil {
		log.Fatal().Err(err).Msg("failed to load config")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	logger := log.Ctx(ctx).With().Interface("server config", cfg.Server).Logger()
	ctx = logger.WithContext(ctx)
	defer cancel()

	rootCmd, err := polygoncmd.RootCmd(ctx, cfg.Port)
	if err != nil {
		log.Fatal().Err(err).Msg("error when setting up the root cmd for the Polygon CLI")
	}
	if err := rootCmd.Execute(); err != nil {
		log.Fatal().Err(err).Msg("error while executing Polygon CLI")
	}
}
