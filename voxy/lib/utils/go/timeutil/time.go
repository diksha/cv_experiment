// Package timeutil provides utilities for working with time
package timeutil

import (
	"time"
)

// RoundUp rounds a time up to the nearest duration
func RoundUp(t time.Time, d time.Duration) time.Time {
	truncated := t.Truncate(d)
	if truncated.Equal(t) {
		return t
	}

	return truncated.Add(d)
}

// RoundDown rounds a time down to the nearest duration
func RoundDown(t time.Time, d time.Duration) time.Time {
	return t.Truncate(d)
}
