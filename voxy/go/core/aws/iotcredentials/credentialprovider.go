// Package iotcredentials provides a utility for loading credentials for the aws sdk go v2 dynamically from AWS IOT's credentials endpoint
package iotcredentials

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
)

// LoadTLSConfig loads a tls configuration suitable for this credential provider
// from greengrass certificate files
func LoadTLSConfig(certPath, keyPath, rootCAPath string) (*tls.Config, error) {
	cert, err := tls.LoadX509KeyPair(certPath, keyPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tls keypair: %w", err)
	}

	caCert, err := os.ReadFile(rootCAPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load cacert: %w", err)
	}
	caCertPool := x509.NewCertPool()
	caCertPool.AppendCertsFromPEM(caCert)

	return &tls.Config{
		MinVersion:   tls.VersionTLS12,
		Certificates: []tls.Certificate{cert},
		RootCAs:      caCertPool,
	}, nil
}

// IOTEndpointOutput is the format of the output from the iot credentials endpoint
// as documented here: https://docs.aws.amazon.com/iot/latest/developerguide/authorizing-direct-aws.html
type IOTEndpointOutput struct {
	Credentials struct {
		AccessKeyID     string `json:"accessKeyId"`
		SecretAccessKey string `json:"secretAccessKey"`
		SessionToken    string `json:"sessionToken"`
		Expiration      string `json:"expiration"`
	} `json:"credentials"`
}

var _ aws.CredentialsProvider = (*Provider)(nil)

// Option is an option for configuring a Provider
type Option func(*Provider) error

// Provider implements aws.CredentialsProvider
type Provider struct {
	client    *http.Client
	thingName string
	endpoint  string
}

// Retrieve fetches credentials from the configured AWS IOT Credentials endpoint
func (p *Provider) Retrieve(ctx context.Context) (aws.Credentials, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.endpoint, nil)
	if err != nil {
		return aws.Credentials{}, fmt.Errorf("failed to construct iot credentials request: %w", err)
	}
	req.Header.Add("x-amzn-iot-thingname", p.thingName)

	resp, err := p.client.Do(req)
	if err != nil {
		return aws.Credentials{}, fmt.Errorf("iot credentials request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		if err == nil {
			return aws.Credentials{}, fmt.Errorf("iot credentials request error status=%d, body=%s", resp.StatusCode, string(body))
		}
		return aws.Credentials{}, fmt.Errorf("iot credentials request error status=%d, failed to read body", resp.StatusCode)
	}

	var out IOTEndpointOutput
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return aws.Credentials{}, fmt.Errorf("failed to unmarshal iot credentials request: %w", err)
	}

	expiration, err := time.Parse(time.RFC3339, out.Credentials.Expiration)
	if err != nil {
		return aws.Credentials{}, fmt.Errorf("failed to parse iot credentials expiration: %w", err)
	}

	return aws.Credentials{
		AccessKeyID:     out.Credentials.AccessKeyID,
		SecretAccessKey: out.Credentials.SecretAccessKey,
		SessionToken:    out.Credentials.SessionToken,
		Source:          "IOTCredentialsProvider",
		CanExpire:       true,
		Expires:         expiration,
	}, nil
}

// NewProvider constructs a new IOT credentials provider from the passed in configuration
func NewProvider(endpoint, thingName, roleAlias string, opts ...Option) (*Provider, error) {
	provider := &Provider{
		thingName: thingName,
		endpoint:  fmt.Sprintf("https://%s/role-aliases/%s/credentials", endpoint, roleAlias),
	}

	for _, optfn := range opts {
		if err := optfn(provider); err != nil {
			return nil, fmt.Errorf("failed to set provider option: %w", err)
		}
	}

	if provider.client == nil {
		provider.client = &http.Client{}
	}

	return provider, nil
}

// WithHTTPClient sets the http client for a Provider
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) error {
		if p.client != nil {
			return fmt.Errorf("attempted to set http client, but http client already set")
		}
		p.client = client
		return nil
	}
}

// WithTLSConfig sets the tls configuration for the Provider
func WithTLSConfig(tlsConfig *tls.Config) Option {
	return func(p *Provider) error {
		if p.client == nil {
			p.client = &http.Client{}
		}

		if p.client.Transport == nil {
			tr := http.DefaultTransport.(*http.Transport).Clone()
			tr.TLSClientConfig = tlsConfig
			p.client.Transport = tr
		} else if tr, ok := p.client.Transport.(*http.Transport); ok {
			if tr.TLSClientConfig != nil {
				return fmt.Errorf("attempted to set tls config but tls config already set")
			}
			tr.TLSClientConfig = tlsConfig
		} else {
			return fmt.Errorf("incompatible transport, unable to set tls config")
		}
		return nil
	}
}

// WithCertificates sets the tls configuration for the provier from the passed in certificate paths
func WithCertificates(certPath, keyPath, rootCAPath string) Option {
	return func(p *Provider) error {
		tlsConfig, err := LoadTLSConfig(certPath, keyPath, rootCAPath)
		if err != nil {
			return fmt.Errorf("failed to load certificates: %w", err)
		}

		err = WithTLSConfig(tlsConfig)(p)
		if err != nil {
			return fmt.Errorf("failed to set tls config for passed in certificates: %w", err)
		}

		return nil
	}
}
