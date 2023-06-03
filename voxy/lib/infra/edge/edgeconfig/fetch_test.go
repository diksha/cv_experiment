package edgeconfig_test

import (
	"context"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/secretsmanager"
	"github.com/aws/aws-sdk-go-v2/service/secretsmanager/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/lib/infra/edge/edgeconfig"
	"github.com/voxel-ai/voxel/lib/utils/aws/fake"
)

func TestFromAWS(t *testing.T) {
	testdata := mustReadTestdata(t, "testdata/edgeconfig.yaml")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	fake := new(fake.SecretsManagerClient)
	fake.ListSecretsReturns(&secretsmanager.ListSecretsOutput{
		SecretList: []types.SecretListEntry{{
			Name: aws.String("edgeconfig.yaml"),
			ARN:  aws.String("some-fake-secret-arn"),
			Tags: []types.Tag{{Key: aws.String("edge:allowed-uuid:primary"), Value: aws.String("some-fake-uuid")}},
		},
		},
	}, nil)
	fake.GetSecretValueReturns(&secretsmanager.GetSecretValueOutput{
		SecretString: aws.String(string(testdata)),
	}, nil)

	cfg, err := edgeconfig.FromAWS(ctx, fake, "some-fake-uuid")
	require.NoError(t, err, "should succeed to fetch an edgeconfig from aws")
	assert.Len(t, cfg.Streams, 1, "edgeconfig.yaml should have one stream")
}
