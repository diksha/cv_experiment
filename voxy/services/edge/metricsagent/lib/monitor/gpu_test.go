package monitor

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPublishGPUUsageQuicksync(t *testing.T) {
	reader, err := os.Open(filepath.Join("testdata", "intel_top_gpu.out"))
	if err != nil {
		t.Fatal(err)
	}

	statsChannel := make(chan intelGPUTopStats)
	go func() {
		err := parseGPUQuicksyncMetrics(reader, statsChannel)
		assert.NoError(t, err, "calling parseGPUQuicksyncMetrics should not cause an error")
	}()

	expectedRenderUsage := []float64{
		0.0,
		12.904720,
		12.479521,
		12.947725,
	}

	expectedVideoUsage := []float64{
		100.0,
		25.83219,
		23.877657,
		22.746993,
	}

	actualRenderUsage := []float64{}
	actualVideoUsage := []float64{}
	for stats := range statsChannel {
		actualRenderUsage = append(actualRenderUsage, stats.Engines["Render/3D/0"].Busy)
		actualVideoUsage = append(actualVideoUsage, stats.Engines["Video/0"].Busy)
	}

	assert.EqualValues(t, expectedRenderUsage, actualRenderUsage)
	assert.EqualValues(t, expectedVideoUsage, actualVideoUsage)
}
