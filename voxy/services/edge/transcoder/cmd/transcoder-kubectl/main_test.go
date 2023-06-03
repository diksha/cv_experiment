package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsValidImageName(t *testing.T) {
	valid := []string{
		"360054435465.dkr.ecr.us-west-2.amazonaws.com/voxel/edge/edge-transcoder-quicksync:voxel.edge.QuicksyncTranscoderDev-0.0.21",
		"360054435465.dkr.ecr.us-west-2.amazonaws.com/voxel/edge/edge-transcoder-quicksync:b09fc3502c5a",
		"registry/rabbit",
		"registry/rabbit:3",
		"rabbit",
		"rabbit",
		"registry.example.com/org/image-name",
		"registry/org/image-name",
		"registry/image-name",
		"image-name",
		"registry.example.com/org/image-name:version",
		"registry/org/image-name:version",
		"registry/image-name:version",
		"image-name:version",
		"ubuntu@sha256:45b23dee08af5e43a7fea6c4cf9c25ccf269ee113168c19722f87876677c5cb2",
		"image:v1.1.1-patch",
		"your-domain.com/image/tag:v1.1.1-patch1",
		"123.123.123.123:123/image/tag:v1.0.0",
	}
	invalid := []string{
		"| echo",
		"&& echo",
		"|| echo",
		"|| exec echo",
		"&&echo",
		"\"||echo",
	}

	for _, img := range valid {
		assert.Truef(t, isValidImageName(img), "%q should be a valid image name", img)
	}

	for _, img := range invalid {
		assert.Falsef(t, isValidImageName(img), "%q should not be a valid image name", img)
	}
}
