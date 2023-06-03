package metricsctx_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/voxel-ai/voxel/go/edge/metricsctx"
)

func TestDimensionsClone(t *testing.T) {
	d := make(metricsctx.Dimensions)
	d["foo"] = "bar"

	clone := d.Clone()
	clone["foo"] = "baz"

	assert.Equal(t, "bar", d["foo"], "pre-clone map should be unchanged")
	assert.Equal(t, "baz", clone["foo"], "post-clone map should be changed")
}
