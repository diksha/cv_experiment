package edgeconfig

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/secretsmanager"
	"github.com/aws/aws-sdk-go-v2/service/secretsmanager/types"

	edgeconfigpb "github.com/voxel-ai/voxel/protos/edge/edgeconfig/v1"
)

// ErrNotFound indicates that the requested EdgeConfig was not found
var ErrNotFound = errors.New("edgeconfig not found")

// SecretsManagerAPI is the set of API calls required by this library to fetch
// EdgeConfig secrets from AWS SecretsManager
type SecretsManagerAPI interface {
	ListSecrets(context.Context, *secretsmanager.ListSecretsInput, ...func(*secretsmanager.Options)) (*secretsmanager.ListSecretsOutput, error)
	GetSecretValue(context.Context, *secretsmanager.GetSecretValueInput, ...func(*secretsmanager.Options)) (*secretsmanager.GetSecretValueOutput, error)
}

var _ SecretsManagerAPI = (*secretsmanager.Client)(nil)

const (
	edgeAllowedUUIDPrimaryTag = "edge:allowed-uuid:primary"
	edgeConfigFilename        = "edgeconfig.yaml"
)

func findEdgeConfigSecretARN(ctx context.Context, client SecretsManagerAPI, thingName string) (string, error) {
	resp, err := client.ListSecrets(ctx, &secretsmanager.ListSecretsInput{
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

	return "", fmt.Errorf("failed to find secret arn for %q: %w", thingName, ErrNotFound)
}

func fetchEdgeConfigSecret(ctx context.Context, client SecretsManagerAPI, secretARN string) (string, error) {
	res, err := client.GetSecretValue(ctx, &secretsmanager.GetSecretValueInput{
		SecretId: aws.String(secretARN),
	})
	if err != nil {
		return "", fmt.Errorf("failed to get secret value for %q: %w", secretARN, err)
	}

	return aws.ToString(res.SecretString), nil
}

// FromAWS fetches an edgeconfig secret for the specified UUID from AWS SecretsManager
func FromAWS(ctx context.Context, client SecretsManagerAPI, uuid string) (*edgeconfigpb.EdgeConfig, error) {
	secretARN, err := findEdgeConfigSecretARN(ctx, client, uuid)
	if err != nil {
		return nil, fmt.Errorf("failed to find edgeconfig.yaml secret for edge uuid %q: %w", uuid, err)
	}

	secretValue, err := fetchEdgeConfigSecret(ctx, client, secretARN)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch secret arn %q: %w", secretARN, err)
	}

	edgeConfig, err := ParseYAML(secretValue)
	if err != nil {
		return nil, fmt.Errorf("failed to parse yaml for secret arn %q: %w", secretARN, err)
	}

	return edgeConfig, nil
}
