package main

import (
	"fmt"
	"time"

	edgeconfigpb "github.com/voxel-ai/voxel/protos/edge/edgeconfig/v1"
)

// Config is the set of command line/env/file based configuration parameters this program accepts
type Config struct {
	StreamConfig string `usage:"streamconfig.yaml file path"`

	Smoketest bool `usage:"run a smoketest to make sure transcoding is possible and exit"`
	DebugMode bool `usage:"run the transcoder in a debug configuration with a test input, software encode, and a disabled metrics publisher"`

	AWS    ConfigAWS
	IOT    ConfigIOT
	FFmpeg ConfigFFmpeg
}

// ConfigFFmpeg holds config values passed to ffmpeg
type ConfigFFmpeg struct {
	LogLevel string `usage:"ffmpeg log level, defaults to error if unset" default:""`
}

// ConfigIOT hold config values relating to AWS IOT
type ConfigIOT struct {
	ThingName     string `usage:"AWS IOT Thing Name"`
	CredsEndpoint string `usage:"AWS IOT Credentials Endpoint"`
	Cert          string `usage:"AWS IOT Greengrass Certificate"`
	PrivKey       string `usage:"AWS IOT Greengrass Private Key"`
	RootCA        string `usage:"AWS IOT Greengrass Root CA Certificate"`
	RoleAlias     string `usage:"AWS IOT Greengrass Role Alias"`
}

// ConfigAWS holds AWS configuration values
type ConfigAWS struct {
	Region string `usage:"aws region" default:"us-west-2"`
}

const (
	statsInterval                 = 60 * time.Second
	defaultVideoBitrateKbps       = 500
	defaultSegmentDurationSeconds = 10.0
)

// DefaultStreamConfig is the default set of stream configuration values used by this transcoder
var DefaultStreamConfig = edgeconfigpb.StreamConfig{
	VideoBitrateKbps: defaultVideoBitrateKbps,
	SegmentDurationS: defaultSegmentDurationSeconds,
	Scaler: &edgeconfigpb.StreamConfig_Scaler{
		Enabled:    true,
		Resolution: edgeconfigpb.StreamConfig_Scaler_RESOLUTION_720P,
	},
}

func resolutionPixels(resolution edgeconfigpb.StreamConfig_Scaler_Resolution) (int, error) {
	switch resolution {
	case edgeconfigpb.StreamConfig_Scaler_RESOLUTION_1080P:
		return 1080, nil
	case edgeconfigpb.StreamConfig_Scaler_RESOLUTION_720P:
		return 720, nil
	case edgeconfigpb.StreamConfig_Scaler_RESOLUTION_480P:
		return 480, nil
	}

	return 0, fmt.Errorf("unsupported scaler resolution %v", resolution)
}
