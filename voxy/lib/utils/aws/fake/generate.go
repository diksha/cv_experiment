// Package fake provides client for mocking AWS clients. It is intended to be used
// soley for generating fake clients for unit tests. These clients are not mean to be used
// as dependencies for production code.
package fake

import (
	"context"

	"github.com/aws/aws-sdk-go-v2/service/cloudwatch"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideo"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideoarchivedmedia"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/secretsmanager"
	"github.com/aws/aws-sdk-go-v2/service/sts"
)

//go:generate go run github.com/maxbrunsfeld/counterfeiter/v6 -generate

//counterfeiter:generate --fake-name S3Client -o . . s3API
type s3API interface {
	HeadObject(context.Context, *s3.HeadObjectInput, ...func(*s3.Options)) (*s3.HeadObjectOutput, error)
	PutObject(context.Context, *s3.PutObjectInput, ...func(*s3.Options)) (*s3.PutObjectOutput, error)
	GetObject(context.Context, *s3.GetObjectInput, ...func(*s3.Options)) (*s3.GetObjectOutput, error)
}

var _ s3API = (*s3.Client)(nil)

//counterfeiter:generate --fake-name KinesisVideoClient -o . . kinesisVideoAPI
type kinesisVideoAPI interface {
	GetDataEndpoint(context.Context, *kinesisvideo.GetDataEndpointInput, ...func(*kinesisvideo.Options)) (*kinesisvideo.GetDataEndpointOutput, error)
}

var _ kinesisVideoAPI = (*kinesisvideo.Client)(nil)

//counterfeiter:generate --fake-name KinesisVideoArchivedMediaClient -o . . kinesisVideoArchivedMediaAPI
type kinesisVideoArchivedMediaAPI interface {
	ListFragments(context.Context, *kinesisvideoarchivedmedia.ListFragmentsInput, ...func(*kinesisvideoarchivedmedia.Options)) (*kinesisvideoarchivedmedia.ListFragmentsOutput, error)
	GetMediaForFragmentList(context.Context, *kinesisvideoarchivedmedia.GetMediaForFragmentListInput, ...func(*kinesisvideoarchivedmedia.Options)) (*kinesisvideoarchivedmedia.GetMediaForFragmentListOutput, error)
}

var _ kinesisVideoArchivedMediaAPI = (*kinesisvideoarchivedmedia.Client)(nil)

//counterfeiter:generate --fake-name CloudwatchClient -o . . cloudwatchAPI
type cloudwatchAPI interface {
	PutMetricData(context.Context, *cloudwatch.PutMetricDataInput, ...func(*cloudwatch.Options)) (*cloudwatch.PutMetricDataOutput, error)
}

var _ cloudwatchAPI = (*cloudwatch.Client)(nil)

//counterfeiter:generate --fake-name SecretsManagerClient -o . . secretsManagerAPI
type secretsManagerAPI interface {
	ListSecrets(context.Context, *secretsmanager.ListSecretsInput, ...func(*secretsmanager.Options)) (*secretsmanager.ListSecretsOutput, error)
	GetSecretValue(context.Context, *secretsmanager.GetSecretValueInput, ...func(*secretsmanager.Options)) (*secretsmanager.GetSecretValueOutput, error)
}

var _ secretsManagerAPI = (*secretsmanager.Client)(nil)

//counterfeiter:generate --fake-name STSClient -o . . stsAPI
type stsAPI interface {
	GetCallerIdentity(context.Context, *sts.GetCallerIdentityInput, ...func(*sts.Options)) (*sts.GetCallerIdentityOutput, error)
}

var _ stsAPI = (*sts.Client)(nil)
