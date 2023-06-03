package retry

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

type mockSleeper struct {
	sleeps []time.Duration
}

func (ms *mockSleeper) Sleep(d time.Duration) { ms.sleeps = append(ms.sleeps, d) }

func TestExponential(t *testing.T) {
	clock = &mockSleeper{}
	count := 5
	errors := []error{}
	err := Exponential{
		Initial:  time.Duration(1),
		MaxDelay: time.Duration(10),
		Log: func(err error) {
			errors = append(errors, err)
		},
	}.Do(context.Background(), func(ctx context.Context) error {
		if count == 0 {
			return nil
		}
		count--
		return fmt.Errorf("not done yet")
	})
	assert.NoError(t, err, "ends with no error")
	assert.Len(t, clock.(*mockSleeper).sleeps, 5, "should sleep 5 times")
	assert.Equal(t, []time.Duration{1, 2, 4, 8, 10}, clock.(*mockSleeper).sleeps, "sleeps should be correct")
}

func TestExponentialError(t *testing.T) {
	clock = &mockSleeper{}
	count := 5
	ctx, cancel := context.WithCancel(context.Background())
	err := Exponential{
		Initial:  time.Duration(1),
		MaxDelay: time.Duration(10),
	}.Do(ctx, func(ctx context.Context) error {
		if count == 0 {
			cancel()
		}
		count--
		return fmt.Errorf("not done yet")
	})
	assert.Error(t, err, "ends with an error")
	assert.Equal(t, []time.Duration{1, 2, 4, 8, 10, 10}, clock.(*mockSleeper).sleeps, "sleeps should be correct")
}
