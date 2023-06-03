package exampleproto_test

import (
	"testing"

	"google.golang.org/protobuf/encoding/protojson"

	"github.com/voxel-ai/voxel/protos/examplepb/v1"
)

func TestExampleProto(t *testing.T) {
	ex := &examplepb.Example{
		Message: "example message",
	}
	t.Log(protojson.Format(ex))
}
