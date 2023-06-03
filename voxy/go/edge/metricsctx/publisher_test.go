package metricsctx_test

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/cloudwatch"
	"github.com/aws/aws-sdk-go-v2/service/cloudwatch/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/go/edge/metricsctx"
)

type mockCloudwatch struct {
	inputs []*cloudwatch.PutMetricDataInput
}

func (m *mockCloudwatch) PutMetricData(ctx context.Context, input *cloudwatch.PutMetricDataInput, optFns ...func(*cloudwatch.Options)) (*cloudwatch.PutMetricDataOutput, error) {
	m.inputs = append(m.inputs, input)
	return &cloudwatch.PutMetricDataOutput{}, nil
}

func TestPublisher(t *testing.T) {
	mcw := &mockCloudwatch{}

	metricts := time.Now()

	func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		ctx = metricsctx.WithPublisher(ctx, metricsctx.Config{
			Client:     mcw,
			Interval:   1 * time.Millisecond,
			Namespace:  "testnamespace",
			Dimensions: metricsctx.Dimensions{"fooa": "bar"},
		})

		metricsctx.Publish(ctx, "testmetric", metricts, metricsctx.UnitBytes, 1.5, metricsctx.Dimensions{"foob": "baz"})

		time.Sleep(100 * time.Millisecond)

		metricsctx.Publish(ctx, "testmetric", time.Now(), metricsctx.UnitBytes, 1.5, metricsctx.Dimensions{"foob": "baz"})

		time.Sleep(100 * time.Millisecond)
	}()

	require.Len(t, mcw.inputs, 2, "should receive two put metric data calls")
	require.Len(t, mcw.inputs[0].MetricData, 1, "should receive one datapoint in the first input")

	assert.Equal(t, "testnamespace", *mcw.inputs[0].Namespace, "namespace should be correct")
	datum := mcw.inputs[0].MetricData[0]
	assert.Equal(t, "testmetric", *datum.MetricName, "metric name should be correct")
	assert.Equal(t, metricts, *datum.Timestamp, "timestamp should be correct")
	assert.Equal(t, types.StandardUnitBytes, datum.Unit, "unit should be correct")
	assert.EqualValues(t, 1.5, *datum.Value, "value should be correct")
	assert.EqualValues(t, 1, *datum.StorageResolution, "storage resolution should be correct")
}

func TestNoNaNInf(t *testing.T) {
	// test that NaN and Inf values are not sent to cloudwatch
	mcw := &mockCloudwatch{}

	metricts := time.Now()

	func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		ctx = metricsctx.WithPublisher(ctx, metricsctx.Config{
			Client:     mcw,
			Interval:   1 * time.Millisecond,
			Namespace:  "testnamespace",
			Dimensions: metricsctx.Dimensions{"fooa": "bar"},
		})

		metricsctx.Publish(ctx, "testmetric", metricts, metricsctx.UnitNone, 0.0, nil)
		time.Sleep(10 * time.Millisecond)

		metricsctx.Publish(ctx, "testmetric", metricts, metricsctx.UnitNone, math.NaN(), nil)
		metricsctx.Publish(ctx, "testmetric", metricts, metricsctx.UnitNone, math.Inf(-1), nil)
		metricsctx.Publish(ctx, "testmetric", metricts, metricsctx.UnitNone, math.Inf(1), nil)
		time.Sleep(10 * time.Millisecond)

		metricsctx.Publish(ctx, "testmetric", metricts, metricsctx.UnitNone, 1.0, nil)
		time.Sleep(10 * time.Millisecond)
	}()

	require.Len(t, mcw.inputs, 2, "should receive two put metric data calls")

	require.Len(t, mcw.inputs[0].MetricData, 1, "should receive one datapoint in the first input")
	assert.Equal(t, 0.0, *mcw.inputs[0].MetricData[0].Value, "value should be correct")

	require.Len(t, mcw.inputs[1].MetricData, 1, "should receive one datapoint in the second input")
	assert.Equal(t, 1.0, *mcw.inputs[1].MetricData[0].Value, "value should be correct")
}

func convertDimensionsToMap(dimensions []types.Dimension) map[string]string {
	dimensionMap := map[string]string{}
	for _, dim := range dimensions {
		dimensionMap[*dim.Name] = *dim.Value
	}
	return dimensionMap
}

func TestDimensionHandling(t *testing.T) {
	mcw := &mockCloudwatch{}

	metricts := time.Now()

	func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		ctx = metricsctx.WithPublisher(ctx, metricsctx.Config{
			Client:     mcw,
			Interval:   1 * time.Millisecond,
			Namespace:  "testnamespace",
			Dimensions: metricsctx.Dimensions{"fooa": "bar"},
		})

		// test adding dimension
		metricsctx.Publish(ctx, "testmetric", metricts, metricsctx.UnitBytes, 1.5, metricsctx.Dimensions{"foob": "baz"})
		time.Sleep(100 * time.Millisecond)

		// test only default dimension
		metricsctx.Publish(ctx, "testmetric", metricts, metricsctx.UnitBytes, 1.5, nil)
		time.Sleep(100 * time.Millisecond)

		// test override dimension
		metricsctx.Publish(ctx, "testmetric", time.Now(), metricsctx.UnitBytes, 1.5, metricsctx.Dimensions{"fooa": "bat"})
		time.Sleep(100 * time.Millisecond)
	}()

	require.Len(t, mcw.inputs, 3, "should receive two put metric data calls")

	metric1 := mcw.inputs[0].MetricData[0]
	metric2 := mcw.inputs[1].MetricData[0]
	metric3 := mcw.inputs[2].MetricData[0]

	metric1Dimensions := convertDimensionsToMap(metric1.Dimensions)
	metric2Dimensions := convertDimensionsToMap(metric2.Dimensions)
	metric3Dimensions := convertDimensionsToMap(metric3.Dimensions)

	assert.Len(t, metric1.Dimensions, 2, "metric #1 should have correct number of dimensions")
	assert.Contains(t, metric1Dimensions, "fooa", "metric #1 should contain default dimension")
	assert.Contains(t, metric1Dimensions, "foob", "metric #1 should contain added dimension")
	assert.Equal(t, metric1Dimensions["fooa"], "bar", "metric #1 default dimension should have correct value")
	assert.Equal(t, metric1Dimensions["foob"], "baz", "metric #1 added dimension should have correct value")

	assert.Len(t, metric2.Dimensions, 1, "metric #2 should have correct number of dimensions")
	assert.Contains(t, metric2Dimensions, "fooa", "metric #2 should contain default dimension")
	assert.Equal(t, metric2Dimensions["fooa"], "bar", "metric #2 default dimension should have correct value")

	assert.Len(t, metric3.Dimensions, 1, "metric #3 should have correct number of dimensions")
	assert.Contains(t, metric3Dimensions, "fooa", "metric #3 should contain default dimension")
	assert.Equal(t, metric3Dimensions["fooa"], "bat", "metric #3 default dimension should have correct value")
}
