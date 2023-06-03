package fish2persp_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	edgeconfigpb "github.com/voxel-ai/voxel/protos/edge/edgeconfig/v1"
	"github.com/voxel-ai/voxel/services/edge/transcoder/lib/fish2persp"
)

// the checksums below were generated with 400x400 sample jpeg using the following command:
//
// fish2persp -w 400 -h 400 -r 200 -c 200 200 -s 180 -x 0 -y 0 -z 0 -f sample.jpg
const (
	xpgmSum = "90361ded82df98478765a36bf253e89eb35ec060429e630e70d92a26d875e022"
	ypgmSum = "581ea9d36ccc9db18872f09aa68cc8dfb757d6f368ca39fe6282d173b2ff666f"
)

var fish2perspTestConfig = &edgeconfigpb.Fish2PerspRemap{
	Fish: &edgeconfigpb.Fish2PerspRemap_Fish{
		WidthPixels:   400,
		HeightPixels:  400,
		CenterXPixels: 200,
		CenterYPixels: 200,
		RadiusXPixels: 200,
		FovDegrees:    180,
		TiltDegrees:   0,
		RollDegrees:   0,
		PanDegrees:    0,
	},
	Persp: &edgeconfigpb.Fish2PerspRemap_Persp{
		WidthPixels:  400,
		HeightPixels: 400,
		FovDegrees:   100,
	},
}

func sha256sum(t *testing.T, data []byte) string {
	hasher := sha256.New()
	_, err := hasher.Write(data)
	require.NoError(t, err, "must hash bytes")
	return hex.EncodeToString(hasher.Sum(nil))
}

func TestGeneratePGM(t *testing.T) {
	pgms, err := fish2persp.GenerateRemapPGM(context.TODO(), fish2perspTestConfig)
	require.NoError(t, err, "must successfully generate remap pgms")
	t.Logf("pgmx len=%d", len(pgms.X))
	t.Logf("pgmy len=%d", len(pgms.Y))
	assert.Equal(t, xpgmSum, sha256sum(t, pgms.X), "pgm x sha256 must match")
	assert.Equal(t, ypgmSum, sha256sum(t, pgms.Y), "pgm y sha256 must match")
}
