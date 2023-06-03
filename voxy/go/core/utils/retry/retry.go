package retry

import (
	"context"
	"time"
)

type sleeper interface {
	Sleep(time.Duration)
}

type timeSleeper struct{}

func (ts timeSleeper) Sleep(d time.Duration) { time.Sleep(d) }

var clock sleeper = timeSleeper{}

// Exponential is an exponential backoff configuration which
// will wait Initial * 2 ^ C  (where C is the count) between each retry
//
// To cancel a series of retries a context with cancellation should be used
type Exponential struct {
	// Initial specifies the initial backoff delay
	Initial time.Duration

	// If MaxDelay is nonzero, the delay between requests will be capped to this
	MaxDelay time.Duration

	Log func(error)
}

// Do performs Exponential retries
func (e Exponential) Do(ctx context.Context, fn func(context.Context) error) error {
	delay := e.Initial
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		err := func() error {
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			return fn(ctx)
		}()
		if err != nil {
			// log the error if we are retrying
			if e.Log != nil {
				e.Log(err)
			}

			clock.Sleep(delay)

			delay = delay * 2
			if e.MaxDelay != 0 && delay > e.MaxDelay {
				delay = e.MaxDelay
			}
		} else {
			return nil
		}
	}
}
