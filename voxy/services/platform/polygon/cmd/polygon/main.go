package main

import (
	"context"
	"crypto/tls"
	"database/sql"
	"fmt"
	"net"
	"os"
	"os/signal"
	"strconv"
	"syscall"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/cristalhq/aconfig"
	grpczerolog "github.com/grpc-ecosystem/go-grpc-middleware/providers/zerolog/v2"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/reflection"

	"github.com/voxel-ai/voxel/lib/infra/autocertfetcher"
	"github.com/voxel-ai/voxel/lib/utils/healthcheck"
	polygonpb "github.com/voxel-ai/voxel/protos/platform/polygon/v1"
	polygonserver "github.com/voxel-ai/voxel/services/platform/polygon/lib/server"

	"github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors/logging"
)

func getGrpcServerOpts(ctx context.Context, cfg polygonserver.Config) ([]grpc.ServerOption, error) {
	logger := log.Ctx(ctx)
	// Enable the log interceptor
	opts := []grpc.ServerOption{
		grpc.UnaryInterceptor(
			logging.UnaryServerInterceptor(grpczerolog.InterceptorLogger(*logger)),
		),
		grpc.StreamInterceptor(
			logging.StreamServerInterceptor(grpczerolog.InterceptorLogger(*logger)),
		),
	}

	// skip tls setup for dev environment
	if cfg.Environment == "development" {
		logger.Warn().Msg("running in insecure dev mode")
		return opts, nil
	}

	fetcher := &autocertfetcher.Fetcher{}
	if err := fetcher.Init(); err != nil {
		return nil, fmt.Errorf("failed to initialize cert fetcher: %w", err)
	}

	go func() {
		err := fetcher.Run(ctx)
		logger.Fatal().Err(err).Msg("cert fetcher exited")
	}()

	tlsConfig := &tls.Config{
		ClientAuth:               tls.RequireAndVerifyClientCert,
		ClientCAs:                fetcher.GetRootCAs(),
		MinVersion:               tls.VersionTLS12,
		CurvePreferences:         []tls.CurveID{tls.CurveP521, tls.CurveP256, tls.CurveP384},
		PreferServerCipherSuites: true,
		CipherSuites: []uint16{
			tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
			tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
		},
		GetCertificate: fetcher.GetCertificate,
	}
	opts = append(opts, grpc.Creds(credentials.NewTLS(tlsConfig)))

	return opts, nil
}

// ListenAndServeAPI serves the main grpc api of this program
func ListenAndServeAPI(ctx context.Context, cfg polygonserver.Config) error {
	opts, err := getGrpcServerOpts(ctx, cfg)
	if err != nil {
		return fmt.Errorf("failed to start grpc server: %w", err)
	}

	addr := net.JoinHostPort("", strconv.Itoa(cfg.ServicePort))
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("failed to listen: %w", err)
	}

	db, err := sql.Open("postgres", fmt.Sprintf("user=%s password=%s dbname=%s sslmode=%s", cfg.User, cfg.Password, cfg.DatabaseName, cfg.SslMode))
	if err != nil {
		return fmt.Errorf("failed to open db connenction: %w", err)
	}
	cfg.Database = db

	// trunk-ignore(semgrep/go.grpc.security.grpc-server-insecure-connection.grpc-server-insecure-connection): false positive
	grpcServer := grpc.NewServer(opts...)
	awsConfig, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return fmt.Errorf("failed to get AWS config: %w", err)
	}
	s3Client := s3.NewFromConfig(awsConfig)
	srv, err := polygonserver.NewServer(s3Client, cfg)
	if err != nil {
		return fmt.Errorf("failed to make Polygon server: %w", err)
	}
	polygonpb.RegisterPolygonServiceServer(grpcServer, srv)
	reflection.Register(grpcServer)

	go func() {
		sigint := make(chan os.Signal, 2)
		signal.Notify(sigint, os.Interrupt)
		signal.Notify(sigint, syscall.SIGTERM)
		<-sigint

		go func() {
			<-sigint
			log.Fatal().Msg("sigterm received during shutdown attempt")
		}()

		log.Info().Msg("sigterm recieved, about to attempt a GracefulStop()")
		grpcServer.GracefulStop()
		log.Info().Msg("GracefulStop() completed, about to shut down")
		os.Exit(0)
	}()

	if err := grpcServer.Serve(lis); err != nil {
		return fmt.Errorf("grpc server error: %w", err)
	}

	return nil
}

func main() {
	// construct an initial logger
	logger := zerolog.New(os.Stderr).With().Timestamp().Logger()

	var cfg polygonserver.Config
	loader := aconfig.LoaderFor(&cfg, aconfig.Config{})
	if err := loader.Load(); err != nil {
		logger.Fatal().Err(err).Msg("failed to load config")
	}

	if cfg.Environment == "development" {
		logger = zerolog.New(zerolog.ConsoleWriter{Out: os.Stderr}).With().Timestamp().Logger()
	}

	logger.Info().Interface("config", cfg).Msg("loaded configuration")

	ctx := logger.WithContext(context.Background())

	go healthcheck.ListenAndServe(ctx, net.JoinHostPort("", strconv.Itoa(cfg.HealthPort)))

	err := ListenAndServeAPI(ctx, cfg)
	logger.Fatal().Err(err).Msg("api server exited")
}
