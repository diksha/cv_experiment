package devcert

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/sts"
	"google.golang.org/grpc"

	"github.com/voxel-ai/voxel/lib/utils/x509pem"
	devcertpb "github.com/voxel-ai/voxel/protos/platform/devcert/v1"
)

//go:generate go run github.com/maxbrunsfeld/counterfeiter/v6 . FetcherClient

type FetcherClient interface {
	GetDevCert(context.Context, *devcertpb.GetDevCertRequest, ...grpc.CallOption) (*devcertpb.GetDevCertResponse, error)
}

// STS is the subset of the AWS STS functions required by the Fetcher
type STS interface {
	GetCallerIdentity(context.Context, *sts.GetCallerIdentityInput, ...func(*sts.Options)) (*sts.GetCallerIdentityOutput, error)
}

type Fetcher struct {
	Client FetcherClient
	STS    STS
}

func (f *Fetcher) getUsername(ctx context.Context) (string, error) {
	stsClient := f.STS
	if stsClient == nil {
		cfg, err := config.LoadDefaultConfig(ctx)
		if err != nil {
			return "", fmt.Errorf("failed to get username: %w", err)
		}
		stsClient = sts.NewFromConfig(cfg)
	}

	resp, err := stsClient.GetCallerIdentity(ctx, &sts.GetCallerIdentityInput{})
	if err != nil {
		return "", fmt.Errorf("failed to get aws credentials")
	}

	userId := aws.ToString(resp.UserId)
	vals := strings.Split(userId, ":")
	if len(vals) != 2 {
		return "", fmt.Errorf("got invalid username %q from sts", userId)
	}

	return vals[1], nil
}

func CreateRequest(username string) (req *devcertpb.GetDevCertRequest, keyPEM []byte, err error) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to generate private key")
	}

	csr, err := x509.CreateCertificateRequest(rand.Reader, &x509.CertificateRequest{
		Subject: pkix.Name{CommonName: username},
	}, key)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to generate csr: %w", err)
	}

	csrPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE REQUEST",
		Bytes: csr,
	})

	req = &devcertpb.GetDevCertRequest{
		Csr: string(csrPEM),
	}

	keyPEM, err = x509pem.Encode(key)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to build cert request: %w", err)
	}

	return req, keyPEM, nil
}

func (f *Fetcher) Fetch(ctx context.Context) (*Cert, error) {
	username, err := f.getUsername(ctx)
	if err != nil {
		return nil, fmt.Errorf("could not get username for cert request: %w", err)
	}

	req, keyPEM, err := CreateRequest(username)
	if err != nil {
		return nil, fmt.Errorf("failed to get credentials: %w", err)
	}

	client := f.Client
	if client == nil {
		client = &Client{}
	}
	resp, err := client.GetDevCert(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	return &Cert{
		RootCA: []byte(resp.RootCa),
		Cert:   []byte(resp.Cert),
		Key:    keyPEM,
	}, nil
}

func Fetch(ctx context.Context) (*Cert, error) {
	return (&Fetcher{}).Fetch(ctx)
}
