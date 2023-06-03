// Package polygoncmd sets up cobra CLI to accept flags and make requests to the server
package polygoncmd

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"os"

	"github.com/rs/zerolog/log"
	"github.com/spf13/cobra"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"

	polygonclient "github.com/voxel-ai/voxel/protos/platform/polygon/v1"

	polygon "github.com/voxel-ai/voxel/protos/platform/polygon/v1"
	"github.com/voxel-ai/voxel/services/platform/devcert"
)

func getClient(ctx context.Context, port int, runInsecure bool) (polygon.PolygonServiceClient, *grpc.ClientConn, error) {
	defaultAddr := fmt.Sprintf("localhost:%d", port)
	if runInsecure {
		conn, err := grpc.Dial(defaultAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			return nil, nil, fmt.Errorf("failed to connect: %w", err)
		}
		client := polygonclient.NewPolygonServiceClient(conn)
		return client, conn, nil
	}

	cert, err := devcert.Fetch(ctx)
	if err != nil {
		log.Fatal().Err(err).Msg("failed to get devcert")
	}

	certificate, err := tls.X509KeyPair(cert.Cert, cert.Key)
	if err != nil {
		log.Fatal().Err(err).Msg("Load client certification failed")
	}

	capool := x509.NewCertPool()
	if !capool.AppendCertsFromPEM(cert.RootCA) {
		log.Fatal().Err(err).Msg("invalid CA")
	}

	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{certificate},
		MinVersion:   tls.VersionTLS13,
		RootCAs:      capool,
	}
	conn, err := grpc.Dial(defaultAddr, grpc.WithTransportCredentials(credentials.NewTLS(tlsConfig)))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to connect: %w", err)
	}
	client := polygonclient.NewPolygonServiceClient(conn)
	return client, conn, nil
}

func getConfig(ctx context.Context, cameraUUID string, port int, runInsecure bool) (response *polygon.GetGraphConfigYAMLResponse, error error) {
	client, conn, err := getClient(ctx, port, runInsecure)
	if err != nil {
		log.Fatal().Err(err).Msg("failed to get the GRPC client")
	}
	defer func() {
		err = conn.Close()
		if err != nil {
			log.Fatal().Err(err).Msg("could not close the connection")
		}
	}()

	resp, err := client.GetGraphConfigYAML(ctx, &polygon.GetGraphConfigYAMLRequest{CameraUuid: cameraUUID})
	if err != nil {
		return nil, fmt.Errorf("server error when running GetConfig: %w", err)
	}

	return resp, nil
}

func putConfig(ctx context.Context, localOverridePath string, cameraUUID string, reason string, port int, runInsecure bool) (response *polygon.PutGraphConfigYAMLResponse, error error) {
	client, conn, err := getClient(ctx, port, runInsecure)
	if err != nil {
		log.Fatal().Err(err).Msg("failed to get the GRPC client")
	}
	defer func() {
		err = conn.Close()
		if err != nil {
			log.Fatal().Err(err).Msg("could not close the connection")
		}
	}()

	overrideBytes, err := os.ReadFile(localOverridePath)
	if err != nil {
		return nil, fmt.Errorf("error trying to read local override file: %w", err)
	}

	resp, err := client.PutGraphConfigYAML(ctx, &polygon.PutGraphConfigYAMLRequest{
		GraphConfig: string(overrideBytes),
		CameraUuid:  cameraUUID,
		Reason:      reason,
	})
	if err != nil {
		return nil, fmt.Errorf("server error when running PutConfig: %w", err)
	}

	return resp, nil
}

func generateGetConfigCmd(port int) *cobra.Command {
	return &cobra.Command{
		Use:     "get-graph-config-override",
		Aliases: []string{"get"},
		Short:   "Get an override from S3",
		Run: func(cmd *cobra.Command, args []string) {
			cameraUUID, err := cmd.Flags().GetString("camera-uuid")
			if err != nil {
				log.Fatal().Err(err).Msg("Error getting CLI flag camera-uuid")
			}
			runInsecure, err := cmd.Flags().GetBool("insecure")
			if err != nil {
				log.Fatal().Err(err).Msg("Error getting CLI flag insecure")
			}

			res, err := getConfig(cmd.Context(), cameraUUID, port, runInsecure)
			if err != nil {
				log.Fatal().Err(err).Msg("Error when running getConfig")
			}
			log.Info().Msg(res.String())
		},
	}
}

func generatePutConfigCmd(port int) *cobra.Command {
	return &cobra.Command{
		Use:     "put-graph-config-override",
		Aliases: []string{"put"},
		Short:   "Upload an override to S3",
		Run: func(cmd *cobra.Command, args []string) {
			runInsecure, err := cmd.Flags().GetBool("insecure")
			if err != nil {
				log.Fatal().Err(err).Msg("Error getting CLI flag insecure")
			}
			localOverridePath, err := cmd.Flags().GetString("override-path")
			if err != nil {
				log.Fatal().Err(err).Msg("Error getting CLI flag override-path")
			}
			cameraUUID, err := cmd.Flags().GetString("camera-uuid")
			if err != nil {
				log.Fatal().Err(err).Msg("Error getting CLI flag camera-uuid")
			}
			reason, err := cmd.Flags().GetString("reason")
			if err != nil {
				log.Fatal().Err(err).Msg("Error getting CLI flag reason")
			}

			res, err := putConfig(cmd.Context(), localOverridePath, cameraUUID, reason, port, runInsecure)
			if err != nil {
				log.Fatal().Err(err).Msg("Error when running putConfig")
			}
			log.Info().Msg(res.String())
		},
	}
}

func generateRootCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "polygon-cli",
		Short: "Polygon CLI to push and get overrides from S3",
		Long:  "Polygon CLI to push and get overrides from S3",
		Run:   func(cmd *cobra.Command, args []string) {},
	}
}

// RootCmd is the entrypoint to this file and sets up cobra CLI
func RootCmd(polygonContext context.Context, port int) (*cobra.Command, error) {
	rootCmd := generateRootCmd()
	putConfigCmd := generatePutConfigCmd(port)
	getConfigCmd := generateGetConfigCmd(port)

	rootCmd.AddCommand(getConfigCmd)
	rootCmd.AddCommand(putConfigCmd)
	rootCmd.SetContext(polygonContext)

	putConfigCmd.Flags().BoolP("insecure", "i", false, "run the CLI using a local connection to an insecure dummy server")
	putConfigCmd.PersistentFlags().StringP("camera-uuid", "c", "", "UUID (path) of camera")
	err := putConfigCmd.MarkPersistentFlagRequired("camera-uuid")
	if err != nil {
		return nil, fmt.Errorf("error marking reason camera-uuid as required: %w", err)
	}
	putConfigCmd.PersistentFlags().StringP("override-path", "o", "", "path to your local override file")
	err = putConfigCmd.MarkPersistentFlagRequired("override-path")
	if err != nil {
		return nil, fmt.Errorf("error marking reason override-path as required: %w", err)
	}
	putConfigCmd.PersistentFlags().StringP("reason", "r", "", "reason (or any other note you want to include) for making your change")
	err = putConfigCmd.MarkPersistentFlagRequired("reason")
	if err != nil {
		return nil, fmt.Errorf("error marking reason flag as required: %w", err)
	}

	getConfigCmd.Flags().BoolP("insecure", "i", false, "run the CLI using a local connection to an insecure dummy server")
	getConfigCmd.PersistentFlags().StringP("camera-uuid", "c", "", "UUID (path) of camera")
	err = getConfigCmd.MarkPersistentFlagRequired("camera-uuid")
	if err != nil {
		return nil, fmt.Errorf("error marking camera-uuid flag as required: %w", err)
	}

	return rootCmd, nil
}
