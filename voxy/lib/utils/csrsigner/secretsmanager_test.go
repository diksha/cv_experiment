package csrsigner_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/secretsmanager"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/lib/utils/aws/fake"
	"github.com/voxel-ai/voxel/lib/utils/csrsigner"
)

func TestLoadFromSecretsManager(t *testing.T) {
	fakesm := &fake.SecretsManagerClient{}
	ctx := context.Background()

	secretValue, err := json.Marshal(map[string]string{
		"root_ca.crt":         string(testCerts.caCertPEM),
		"intermediate_ca.crt": string(testCerts.signCertPEM),
		"intermediate_ca.key": string(testCerts.signKeyPEM),
		"password":            string(testCerts.signKeyPass),
	})
	require.NoError(t, err, "should marshal secret value")

	fakesm.GetSecretValueReturns(&secretsmanager.GetSecretValueOutput{
		SecretString: aws.String(string(secretValue)),
	}, nil)

	signer, err := csrsigner.LoadFromSecretsManager(ctx, fakesm, "some-fake-secret")
	require.NoError(t, err, "does not error")
	assert.NotNil(t, signer.RootCA, "root ca should not be nil")
	assert.NotNil(t, signer.Certs, "signer CA should be set")
	assert.NotNil(t, signer.Key, "key should be set")
}
