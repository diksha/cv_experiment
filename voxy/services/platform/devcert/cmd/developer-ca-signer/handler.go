// developer-ca-signer can sign certificates for developers to make requests to internal services
package main

import (
	"context"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"strings"

	"github.com/aws/aws-lambda-go/events"
	"github.com/rs/zerolog/log"
	"google.golang.org/protobuf/encoding/protojson"

	"github.com/voxel-ai/voxel/lib/utils/csrsigner"
	"github.com/voxel-ai/voxel/lib/utils/x509pem"
	devcertpb "github.com/voxel-ai/voxel/protos/platform/devcert/v1"
)

// ErrNotAuthorized indicates the user requested a certificate with invalid parameters
var ErrNotAuthorized = errors.New("not authorized")

// Handler is the lambda function handler
type Handler struct {
	Signer *csrsigner.Signer
}

func getCallerID(req *events.LambdaFunctionURLRequest) string {
	if req == nil {
		return ""
	}

	auth := req.RequestContext.Authorizer
	if auth == nil || auth.IAM == nil {
		return ""
	}

	return auth.IAM.CallerID
}

func getUsername(req *events.LambdaFunctionURLRequest) (string, error) {
	callerID := getCallerID(req)
	vals := strings.Split(callerID, ":")
	if len(vals) == 2 {
		return vals[1], nil
	}

	return "", fmt.Errorf("invalid lambda callerId %q", callerID)
}

// Handle is the actual lambda handler code
func (h *Handler) Handle(ctx context.Context, lambdaReq *events.LambdaFunctionURLRequest) (json.RawMessage, error) {
	logger := log.Ctx(ctx)

	req := &devcertpb.GetDevCertRequest{}
	if err := protojson.Unmarshal([]byte(lambdaReq.Body), req); err != nil {
		return nil, fmt.Errorf("failed to unmarshal request: %w", err)
	}

	csr, err := x509pem.Decode[*x509.CertificateRequest]([]byte(req.Csr))
	if err != nil {
		return nil, fmt.Errorf("invalid certificate signing request: %w", err)
	}

	username, err := getUsername(lambdaReq)
	if err != nil {
		logger.Error().Err(err).Msg("authentication failure")
		return nil, ErrNotAuthorized
	}

	logger.Info().
		Str("username", username).
		Str("common_name", csr.Subject.CommonName).
		Msg("got signing request")

	if csr.Subject.CommonName != username {
		logger.Error().
			Str("username", username).
			Str("common_name", csr.Subject.CommonName).
			Msg("username does not match common_name")
		return nil, ErrNotAuthorized
	}

	certs, err := h.Signer.Sign(csr)
	if err != nil {
		return nil, fmt.Errorf("failed to sign request: %w", err)
	}

	certPEM, err := x509pem.Encode(certs)
	if err != nil {
		return nil, fmt.Errorf("failed to encode signed certs: %w", err)
	}

	resp := &devcertpb.GetDevCertResponse{}
	resp.Cert = string(certPEM)
	resp.RootCa = string(pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: h.Signer.RootCA.Raw,
	}))

	out, err := protojson.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal response: %w", err)
	}

	logger.Info().Msg("certificate signing success")

	return json.RawMessage(out), nil
}
