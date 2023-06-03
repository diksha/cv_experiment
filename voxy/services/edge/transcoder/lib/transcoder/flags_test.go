package transcoder_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/voxel-ai/voxel/services/edge/transcoder/lib/transcoder"
)

func TestFlags(t *testing.T) {
	var f transcoder.Flags
	assert.Equal(t, "", f.String(), "zero value should be nil")
	f.Replace("-a")
	assert.Equal(t, "-a", f.String(), "replace should correctly replace items in this list")
	f.Append("-b")
	assert.Equal(t, "-a -b", f.String(), "append should append to this list")
	f.Prepend("-c")
	assert.Equal(t, "-c -a -b", f.String(), "prepend should prepend to this list")
}

func TestSimpleFilter(t *testing.T) {
	var f transcoder.SimpleFilter
	assert.Equal(t, "", f.String(), "zero value is an empty filter")
	f.Replace("a")
	assert.Equal(t, "a", f.String(), "filter replaces correctly")
	f.Append("b")
	assert.Equal(t, "a,b", f.String(), "filter appends correctly")
	f.Prepend("c")
	assert.Equal(t, "c,a,b", f.String(), "filter prepends correctly")
}
