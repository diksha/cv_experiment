package main

import (
	"context"
	"os"
	"time"

	"github.com/cristalhq/aconfig"
	"github.com/davecgh/go-spew/spew"
	"github.com/rs/zerolog"
)

func main() {
	ctx := context.Background()
	logger := zerolog.New(zerolog.ConsoleWriter{
		Out:        os.Stderr,
		NoColor:    true,
		TimeFormat: time.RFC3339,
		PartsOrder: []string{
			zerolog.TimestampFieldName,
			zerolog.LevelFieldName,
			"ctx",
		},
	}).With().Timestamp().Logger()
	ctx = logger.WithContext(ctx)

	spew.Config = spew.ConfigState{
		Indent:                  "    ",
		DisablePointerAddresses: true,
		DisableCapacities:       true,
		SpewKeys:                true,
	}

	// first just attempt to load our main configuration
	var cfg Config
	loader := aconfig.LoaderFor(&cfg, aconfig.Config{})
	if err := loader.Load(); err != nil {
		logger.Fatal().Err(err).Msg("failed to load config")
	}
	logger.Info().Interface("config", cfg).Msg("loaded config")

	// set up and run the app
	app := &App{
		Config: cfg,
	}

	if err := app.Run(ctx); err != nil {
		// try to print any transcoder logs if we have them
		logger.Fatal().Err(err).Msg("transcode ended with error")
	}

	logger.Info().Msg("transcode complete")
}
