package csrsigner

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/secretsmanager"
)

// SecretsManager is the subset of the AWS SecretsManager interface required by LoadFromSecretsManager
type SecretsManager interface {
	GetSecretValue(context.Context, *secretsmanager.GetSecretValueInput, ...func(*secretsmanager.Options)) (*secretsmanager.GetSecretValueOutput, error)
}

// LoadFromSecretsManager attempts to load signing keys from AWS SecretsManager
func LoadFromSecretsManager(ctx context.Context, sm SecretsManager, secretARN string) (*Signer, error) {
	resp, err := sm.GetSecretValue(ctx, &secretsmanager.GetSecretValueInput{
		SecretId: aws.String(secretARN),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to fetch certificate secret from SecretsManager: %w", err)
	}

	var certs struct {
		RootPEM  string `json:"root_ca.crt"`
		CertPEM  string `json:"intermediate_ca.crt"`
		KeyPEM   string `json:"intermediate_ca.key"`
		Password string `json:"password"`
	}

	if err := json.Unmarshal([]byte(aws.ToString(resp.SecretString)), &certs); err != nil {
		return nil, fmt.Errorf("failed to unmarshal certificates: %w", err)
	}

	return LoadFromPEMsWithEncryptedKey([]byte(certs.RootPEM), []byte(certs.CertPEM), []byte(certs.KeyPEM), []byte(certs.Password))
}
