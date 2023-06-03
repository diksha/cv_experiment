package metricsctx

import (
	"context"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/cloudwatch/types"
)

type contextKey int

var (
	dimensionsKey contextKey = 1
	publisherKey  contextKey = 2
)

func publisherFrom(ctx context.Context) chan types.MetricDatum {
	if ch, ok := ctx.Value(publisherKey).(chan types.MetricDatum); ok {
		return ch
	}
	return nil
}

func dimensionsFrom(ctx context.Context) Dimensions {
	if d, ok := ctx.Value(dimensionsKey).(Dimensions); ok {
		return d
	}
	return nil
}

// WithPublisher configures the passed in context with a metrics publisher and starts it.
// Users should be aware that this consumes resources by creating a goroutine, to release them
// just cancel the passed in context and the resources will be released.
func WithPublisher(ctx context.Context, cfg Config) context.Context {
	if cfg.Interval == time.Duration(0) {
		cfg.Interval = 1 * time.Minute
	}
	// default to 1000 metrics in the queue, a relatively arbitrary value
	// extremely high rates of metric publish calls will cause this queue
	// to fill up and the publish call to drop messages
	datach := make(chan types.MetricDatum, 1000)
	ctx = context.WithValue(ctx, publisherKey, datach)
	ctx = WithDimensions(ctx, cfg.Dimensions)

	// start the publisher, which will stop when the passed in context is canceled
	startPublisher(ctx, datach, cfg)

	return ctx
}

// WithDimension overrides a single dimension in the current context's metrics dimension set
func WithDimension(ctx context.Context, name, value string) context.Context {
	return WithDimensions(ctx, Dimensions{name: value})
}

// WithDimensions overrides the curren't context's dimensions with the passed in dimensions
func WithDimensions(ctx context.Context, d Dimensions) context.Context {
	newd := dimensionsFrom(ctx)
	if newd == nil {
		newd = make(Dimensions)
	}

	for k, v := range d {
		newd[k] = v
	}

	return context.WithValue(ctx, dimensionsKey, newd)
}
