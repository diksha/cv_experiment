package metricsctx

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/rs/zerolog/log"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/cloudwatch"
	"github.com/aws/aws-sdk-go-v2/service/cloudwatch/types"
	smithyhttp "github.com/aws/smithy-go/transport/http"
)

const (
	cloudwatchPutMetricDataMaxCount = 1000
	payloadTooLargeStatusCode       = 413

	// Approximately 1 day of receiving 2 metrics per second
	metricDatumMaxBufferCount = 175000
)

// CloudwatchAPI specifies the set of Cloudwatch APIs required by this publishers
type CloudwatchAPI interface {
	PutMetricData(ctx context.Context, input *cloudwatch.PutMetricDataInput, optFns ...func(*cloudwatch.Options)) (*cloudwatch.PutMetricDataOutput, error)
}

var _ CloudwatchAPI = (*cloudwatch.Client)(nil)

// Config holds configuration values for a metric publisher
type Config struct {
	// Client is the cloudwatch client to use, should be set to a *cloudwatch.Client
	Client CloudwatchAPI
	// Interval is the maximum interval between PutMetricData requests, defaults to 1 minute
	Interval time.Duration
	// DryRun allows this metric exporter to run without actually publishing for debugging purposes
	DryRun bool
	// Namespace is the default namespace to use for metrics
	Namespace string
	// Dimensions is an optional set of dimensions to add to all metrics
	Dimensions Dimensions
}

// Unit is the same as the StandardUnit type from github.com/aws/aws-sdk-go-v2/service/cloudwatch/types just duplicated here for ease of use
type Unit types.StandardUnit

// Unit standard values
const (
	UnitSeconds         Unit = "Seconds"
	UnitMicroseconds    Unit = "Microseconds"
	UnitMilliseconds    Unit = "Milliseconds"
	UnitBytes           Unit = "Bytes"
	UnitKilobytes       Unit = "Kilobytes"
	UnitMegabytes       Unit = "Megabytes"
	UnitGigabytes       Unit = "Gigabytes"
	UnitTerabytes       Unit = "Terabytes"
	UnitBits            Unit = "Bits"
	UnitKilobits        Unit = "Kilobits"
	UnitMegabits        Unit = "Megabits"
	UnitGigabits        Unit = "Gigabits"
	UnitTerabits        Unit = "Terabits"
	UnitPercent         Unit = "Percent"
	UnitCount           Unit = "Count"
	UnitBytesSecond     Unit = "Bytes/Second"
	UnitKilobytesSecond Unit = "Kilobytes/Second"
	UnitMegabytesSecond Unit = "Megabytes/Second"
	UnitGigabytesSecond Unit = "Gigabytes/Second"
	UnitTerabytesSecond Unit = "Terabytes/Second"
	UnitBitsSecond      Unit = "Bits/Second"
	UnitKilobitsSecond  Unit = "Kilobits/Second"
	UnitMegabitsSecond  Unit = "Megabits/Second"
	UnitGigabitsSecond  Unit = "Gigabits/Second"
	UnitTerabitsSecond  Unit = "Terabits/Second"
	UnitCountSecond     Unit = "Count/Second"
	UnitNone            Unit = "None"
)

// Publish will make best effort to emit metrics, failures can occur if no metrics logger is configured or if the rate of metric
// publish calls is high enough to overflow the metric data queue
func Publish(ctx context.Context, metricName string, timestamp time.Time, unit Unit, value float64, dimensions Dimensions) {
	if math.IsInf(value, 0) || math.IsNaN(value) {
		log.Warn().
			Str("metricName", metricName).
			Str("unit", string(unit)).
			Float64("value", value).
			Msg("dropping metric due to invalid value")
		return
	}

	datach := publisherFrom(ctx)
	if datach == nil {
		log.Error().
			Str("metricName", metricName).
			Str("unit", string(unit)).
			Float64("value", value).
			Msg("dropping metric due to missing metrics logger")
		return
	}

	allDimensions := dimensionsFrom(ctx).Clone()
	for k, v := range dimensions {
		allDimensions[k] = v
	}

	var dimensionList []types.Dimension
	for k, v := range allDimensions {
		dimensionList = append(dimensionList, types.Dimension{
			Name:  aws.String(k),
			Value: aws.String(v),
		})
	}

	datum := types.MetricDatum{
		MetricName: aws.String(metricName),
		Dimensions: dimensionList,
		// We will set StorageResolution to 1 (1s) making this a high resolution metric
		// Pricing is the same for standard and high resolution metrics, but high resolution
		// alarms do cost more. We may want this to be configurable in the future but for now
		// this seems like a sane default.
		StorageResolution: aws.Int32(1),
		Timestamp:         &timestamp,
		Unit:              types.StandardUnit(unit),
		Value:             aws.Float64(value),
	}

	select {
	case datach <- datum:
	default:
		log.Warn().
			Str("metricName", metricName).
			Str("unit", string(unit)).
			Float64("value", value).
			Msg("dropping metric due to full metric queue")
	}
}

func startPublisher(ctx context.Context, datach chan types.MetricDatum, cfg Config) {
	go func() {
		constantTickerInterval := cfg.Interval
		initialBackoffInterval := time.Second * 1
		currentBackoffInterval := initialBackoffInterval
		inFailState := false

		ticker := time.NewTicker(constantTickerInterval)
		defer ticker.Stop()

		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		buffer := make([]types.MetricDatum, 0, metricDatumMaxBufferCount)
		for {
			select {
			case <-ctx.Done():
				err := ctx.Err()
				if err != nil && !errors.Is(err, context.Canceled) {
					log.Error().Err(err).Msg("cloudwatch metrics publisher stopped with error")
				} else {
					log.Info().Msg("cloudwatch metrics publisher stopped")
				}
				return
			case datum := <-datach:
				if len(buffer) < metricDatumMaxBufferCount {
					buffer = append(buffer, datum)
				} else {
					buffer = buffer[:metricDatumMaxBufferCount]
				}
			case <-ticker.C:
				var err error
				buffer, err = putMetricData(ctx, buffer, cfg)

				if err != nil {
					log.Error().Err(err).Msg("cloudwatch PutMetricData failure")

					// set backoff interval
					if inFailState {
						currentBackoffInterval *= 2
					} else {
						currentBackoffInterval = initialBackoffInterval
					}
					ticker.Reset(currentBackoffInterval)
				} else if inFailState {
					// reset to constant ticker interval
					ticker.Reset(constantTickerInterval)
				}

				inFailState = err != nil
			}
		}
	}()
}

// putMetricData attempts to send one batch of metric data to cloudwatch.
// Returned slice contains the data that was not sent.
// Amount of data sent is limited by number of metrics and payload size of PUT request.
// putMetricData will continue to attempt PUTs on receiving payload-too-large error,
// reducing the number of metrics sent until success or other error is reached.
// On dry run, no data will be sent to cloudwatch and will instead all be printed
func putMetricData(ctx context.Context, data []types.MetricDatum, cfg Config) ([]types.MetricDatum, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	numMetricsToPut := len(data)
	if numMetricsToPut > cloudwatchPutMetricDataMaxCount {
		numMetricsToPut = cloudwatchPutMetricDataMaxCount
	}

	if numMetricsToPut == 0 {
		return nil, nil
	}

	if cfg.DryRun {
		for _, datum := range data {
			fmt.Println(metricDatumString(datum))
		}
		return nil, nil
	}

	for {
		_, err := cfg.Client.PutMetricData(ctx, &cloudwatch.PutMetricDataInput{
			MetricData: data[:numMetricsToPut],
			Namespace:  aws.String(cfg.Namespace),
		})

		if err == nil {
			// remove data which we successfully put
			return data[numMetricsToPut:], nil
		}

		var resError *smithyhttp.ResponseError
		if errors.As(err, &resError) && resError.HTTPStatusCode() == payloadTooLargeStatusCode {
			// Hit limit on POST request size, limit metric data to account for this
			if numMetricsToPut == 1 {
				return data, fmt.Errorf("cloudwatch PutMetricData failure of single metric datum: %w", err)
			}

			log.Info().
				Int("numMetricsToPut", numMetricsToPut).
				Msg("cloudwatch PutMetricData failure due to payload-too-large, halving amount of data transferred...")
			numMetricsToPut /= 2
		} else {
			return data, fmt.Errorf("cloudwatch PutMetricData failure: %w", err)
		}
	}
}

func metricDatumString(datum types.MetricDatum) string {
	return fmt.Sprintf("Metric{Name=%s, Value=%f, Unit=%v}", *datum.MetricName, *datum.Value, datum.Unit)
}
