package devcert_test

import (
	"context"
	"crypto/x509"
	"fmt"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/sts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

	"github.com/voxel-ai/voxel/lib/utils/aws/fake"
	"github.com/voxel-ai/voxel/lib/utils/x509pem"
	devcertpb "github.com/voxel-ai/voxel/protos/platform/devcert/v1"
	"github.com/voxel-ai/voxel/services/platform/devcert"
	"github.com/voxel-ai/voxel/services/platform/devcert/devcertfakes"
)

func TestFetcher(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	devcertclient := &devcertfakes.FakeFetcherClient{}
	stsclient := &fake.STSClient{}

	certdata := []byte("some fake certificate data")
	commonName := "test@voxelai.com"
	stsclient.GetCallerIdentityReturns(&sts.GetCallerIdentityOutput{
		UserId: aws.String(fmt.Sprintf("ABCDEF:%s", commonName)),
	}, nil)

	devcertclient.GetDevCertStub = func(ctx context.Context, req *devcertpb.GetDevCertRequest, opts ...grpc.CallOption) (*devcertpb.GetDevCertResponse, error) {
		certReq, err := x509pem.Decode[*x509.CertificateRequest]([]byte(req.Csr))
		if err != nil {
			return nil, fmt.Errorf("failed to decode pem: %w", err)
		}

		if certReq.Subject.CommonName != commonName {
			return nil, fmt.Errorf("invalid subject common name %q, expected %q", certReq.Subject.CommonName, commonName)
		}

		return &devcertpb.GetDevCertResponse{
			Cert: string(certdata),
		}, nil
	}

	fetcher := &devcert.Fetcher{
		Client: devcertclient,
		STS:    stsclient,
	}

	cert, err := fetcher.Fetch(ctx)
	require.NoError(t, err, "fetcher must not error")
	assert.Equal(t, certdata, cert.Cert, "test cert data must match")

}
