// pinholeserver provides an mTLS terminated proxy server for our edges to
// reach cloud resources more easily from behind a restrictive network
package main

import (
	"context"
	"os"
	"time"

	"github.com/cristalhq/aconfig"
	"github.com/rs/zerolog"
)

// this is a list of domains we allow unauthenticated requests to
// these domains should *only* include the following:
//
//   - AWS IOT Greengrass Credentials endpoint
//   - Pinhole Certificate lambda endpoint
//
// No other domains should be allowed. These are embedded into the code
// because they should almost never change and making them configurable
// might actually create slightly more of a security risk
var unauthenticatedAddrs = []string{
	// production account iot credentials endpoint
	"c1phi7okof0xz.credentials.iot.us-west-2.amazonaws.com:443",
	// pinholecert-production lambda function url
	"7nfdx2t63hwo5cp4jhnsvof7hq0kkcxa.lambda-url.us-west-2.on.aws:443",
}

// Config is all the configuration options available for pinholeserver
type Config struct {
	Addr    string `default:":8443"`
	DevMode bool

	// if TLS is not specified certificate will be autofetched with `autocertfetcher`
	TLS struct {
		RootCA string
		Cert   string
		Key    string
	}
}

func main() {
	ctx := context.Background()
	logger := zerolog.New(zerolog.ConsoleWriter{
		Out:        os.Stderr,
		NoColor:    true,
		TimeFormat: time.RFC3339,
	}).With().Timestamp().Logger()
	ctx = logger.WithContext(ctx)

	var cfg Config
	loader := aconfig.LoaderFor(&cfg, aconfig.Config{})
	if err := loader.Load(); err != nil {
		logger.Fatal().Err(err).Msg("failed to load config")
	}

	app := &App{
		AllowUnauthenticatedAddrs: unauthenticatedAddrs,
		Config:                    cfg,
	}

	if err := app.Listen(ctx); err != nil {
		logger.Fatal().Err(err).Msg("app listener failed")
	}

	if err := app.Serve(ctx); err != nil {
		logger.Fatal().Err(err).Msg("app server failed")
	}

	logger.Info().Msg("app server exited")
}
