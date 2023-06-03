package monitor

import (
	"context"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/cloudwatch/types"
	"github.com/stretchr/testify/assert"
)

func TestLoad(t *testing.T) {
	ctx, cloudWatch := getTestEnvironment()
	ctx, cancel := context.WithTimeout(ctx, 1*time.Second)
	defer cancel()

	expect := []expectedMetric{
		{"Load1", types.StandardUnitNone, 0.0, 100.0},
	}

	go func() {
		err := AverageLoad(ctx, 100*time.Millisecond)
		assert.NoError(t, err, "calling AverageLoad should not cause error")
	}()

	runAssertions(ctx, t, cloudWatch.putChannel, 5, expect)
}

func TestMemoryUsage(t *testing.T) {
	ctx, cloudWatch := getTestEnvironment()
	ctx, cancel := context.WithTimeout(ctx, 1*time.Second)
	defer cancel()

	expect := []expectedMetric{
		{"MemoryUsage", types.StandardUnitPercent, 0.0, 100.0},
	}

	go func() {
		err := MemoryUsage(ctx, 100*time.Millisecond)
		assert.NoError(t, err, "calling MemoryUsage should not cause error")
	}()

	runAssertions(ctx, t, cloudWatch.putChannel, 5, expect)
}

func TestCPUUsage(t *testing.T) {
	ctx, cloudWatch := getTestEnvironment()
	ctx, cancel := context.WithTimeout(ctx, 1*time.Second)
	defer cancel()

	expect := []expectedMetric{
		{"CPUUsage", types.StandardUnitPercent, 0.0, 100.0},
	}

	go func() {
		err := CPUUsage(ctx, 100*time.Millisecond)
		assert.NoError(t, err, "calling CPUUsage should not cause error")
	}()
	runAssertions(ctx, t, cloudWatch.putChannel, 5, expect)
}

func TestDiskUsage(t *testing.T) {
	ctx, cloudWatch := getTestEnvironment()
	ctx, cancel := context.WithTimeout(ctx, 1*time.Second)
	defer cancel()

	expect := []expectedMetric{
		{"MaxPartitionUsage", types.StandardUnitPercent, 0.0, 100.0},
	}

	go func() {
		err := DiskUsage(ctx, 100*time.Millisecond)
		assert.NoError(t, err, "calling DiskUsage should not cause error")
	}()

	runAssertions(ctx, t, cloudWatch.putChannel, 5, expect)
}
