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
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/iot"
	"github.com/rs/zerolog/log"
	"google.golang.org/protobuf/encoding/protojson"

	"github.com/voxel-ai/voxel/lib/utils/csrsigner"
	"github.com/voxel-ai/voxel/lib/utils/x509pem"
	pinholepb "github.com/voxel-ai/voxel/protos/edge/pinhole/v1"
)

// ErrNotAuthorized indicates the requestor was not authorized to have the submitted CSR signed
var ErrNotAuthorized = errors.New("not authorized")

// Handler is a lambda handler compatible with the aws lambda api
type Handler struct {
	Signer    *csrsigner.Signer
	AWSConfig aws.Config
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

func (h *Handler) getEdgeUUID(ctx context.Context, req *events.LambdaFunctionURLRequest) (string, error) {
	callerID := getCallerID(req)
	vals := strings.Split(callerID, ":")
	if len(vals) != 2 {
		return "", fmt.Errorf("invalid caller id string %q", callerID)
	}

	thingName, err := h.getThingNameFromCertificateID(ctx, vals[1])
	if err != nil {
		return "", fmt.Errorf("failed to get thing name from iot certificate id: %w", err)
	}

	return thingName, nil
}

func (h *Handler) getThingNameFromCertificateID(ctx context.Context, certificateID string) (string, error) {
	client := iot.NewFromConfig(h.AWSConfig)
	certResp, err := client.DescribeCertificate(ctx, &iot.DescribeCertificateInput{
		CertificateId: &certificateID,
	})
	if err != nil {
		return "", fmt.Errorf("failed to describe iot certificate %q: %w", certificateID, err)
	}

	thingResp, err := client.ListPrincipalThings(ctx, &iot.ListPrincipalThingsInput{
		Principal: certResp.CertificateDescription.CertificateArn,
	})
	if err != nil {
		return "", fmt.Errorf("failed to list principal things: %w", err)
	}

	if len(thingResp.Things) != 1 {
		return "", fmt.Errorf("expected 1 thing attached to certificate, found %d", len(thingResp.Things))
	}

	return thingResp.Things[0], nil
}

// Handle can correctly handle a function URL request
func (h *Handler) Handle(ctx context.Context, lambdaReq *events.LambdaFunctionURLRequest) (json.RawMessage, error) {
	// add the request id to the log context
	logger := log.Ctx(ctx).With().Str("requestId", lambdaReq.RequestContext.RequestID).Logger()
	ctx = logger.WithContext(ctx)

	req := &pinholepb.SignRequest{}
	if err := protojson.Unmarshal([]byte(lambdaReq.Body), req); err != nil {
		return nil, fmt.Errorf("failed to unmarshal request: %w", err)
	}

	csr, err := x509pem.Decode[*x509.CertificateRequest]([]byte(req.Csr))
	if err != nil {
		return nil, fmt.Errorf("invalid certificate signing request: %w", err)
	}

	edgeUUID, err := h.getEdgeUUID(ctx, lambdaReq)
	if err != nil {
		logger.Error().Err(err).Msg("authentication failure")
		return nil, ErrNotAuthorized
	}

	logger.Info().
		Str("edge_uuid", edgeUUID).
		Str("common_name", csr.Subject.CommonName).
		Msg("got signing request")

	if csr.Subject.CommonName != edgeUUID {
		logger.Error().
			Str("edge_uuid", edgeUUID).
			Str("common_name", csr.Subject.CommonName).
			Msg("edge_uuid does not match common_name")
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

	resp := &pinholepb.SignResponse{}
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
