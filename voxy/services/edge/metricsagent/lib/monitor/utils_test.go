package monitor

import (
	"context"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/cloudwatch"
	"github.com/aws/aws-sdk-go-v2/service/cloudwatch/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/go/edge/metricsctx"
)

type mockCloudWatch struct {
	putChannel chan *cloudwatch.PutMetricDataInput
}

func (m *mockCloudWatch) PutMetricData(ctx context.Context, input *cloudwatch.PutMetricDataInput, optFns ...func(*cloudwatch.Options)) (*cloudwatch.PutMetricDataOutput, error) {
	m.putChannel <- input
	return &cloudwatch.PutMetricDataOutput{}, nil
}

type expectedMetric struct {
	Name       string
	Unit       types.StandardUnit
	LowerBound float64
	UpperBound float64
}

func runAssertions(ctx context.Context, t *testing.T, putChannel chan *cloudwatch.PutMetricDataInput, numMetricsExpected int, expect []expectedMetric) {
	totalMetrics := 0
	hitsPerMetric := make(map[string]int, len(expect))
	metricNameToExpectation := make(map[string]expectedMetric, len(expect))

	for _, expectation := range expect {
		hitsPerMetric[expectation.Name] = 0
		metricNameToExpectation[expectation.Name] = expectation
	}

	for publishIndex := 0; ; publishIndex++ {
		select {
		case data := <-putChannel:
			for _, metric := range data.MetricData {
				totalMetrics++

				hitsPerMetric[*metric.MetricName]++

				expect, ok := metricNameToExpectation[*metric.MetricName]
				require.Truef(t, ok, "metric %q should be expected", *metric.MetricName)

				assert.Equal(t, expect.Unit, metric.Unit, "units should match")
				assert.LessOrEqual(t, expect.LowerBound, *metric.Value, "value should be greater than lower bound")
				assert.GreaterOrEqual(t, expect.UpperBound, *metric.Value, "value should be less than upper bound")
			}

			if totalMetrics >= numMetricsExpected {
				expectedMetricsPerType := numMetricsExpected / len(expect)
				for name, hitCount := range hitsPerMetric {
					assert.GreaterOrEqualf(t, hitCount, expectedMetricsPerType, "should have at least %v metrics of type %q", expectedMetricsPerType, name)
				}
				return
			}

		case <-ctx.Done():
			for name, hitCount := range hitsPerMetric {
				t.Errorf("recieved %v metrics of type %q", hitCount, name)
			}

			t.Fatalf("timed out while waiting for data after %v publishes", publishIndex)
		}
	}
}

func getTestEnvironment() (context.Context, *mockCloudWatch) {
	mcw := &mockCloudWatch{make(chan *cloudwatch.PutMetricDataInput)}
	ctx := metricsctx.WithPublisher(context.Background(), metricsctx.Config{
		Client:     mcw,
		Interval:   1 * time.Millisecond,
		Namespace:  "testnamespace",
		Dimensions: metricsctx.Dimensions{"foo": "bar"},
	})

	return ctx, mcw
}
