package monitor

import (
	"context"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/cloudwatch/types"
	"github.com/stretchr/testify/assert"
)

func TestNetworkStats(t *testing.T) {
	t.Skip("Unable to fetch network interfaces in buildkite environment")

	ctx, cloudWatch := getTestEnvironment()
	ctx, cancel := context.WithTimeout(ctx, 1*time.Second)
	defer cancel()

	expect := []expectedMetric{
		{"NetworkOutKBPS", types.StandardUnitKilobytesSecond, 0, 1000000},
		{"NetworkInKBPS", types.StandardUnitKilobytesSecond, 0, 1000000},
		{"PacketLossIn", types.StandardUnitPercent, 0, 100},
		{"PacketLossOut", types.StandardUnitPercent, 0, 100},
	}

	go func() {
		err := NetworkStats(ctx, 100*time.Millisecond)
		assert.NoError(t, err, "calling NetworkStats should not cause error")
	}()

	runAssertions(ctx, t, cloudWatch.putChannel, 20, expect)
}
