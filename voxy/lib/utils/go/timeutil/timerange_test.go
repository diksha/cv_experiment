package timeutil_test

import (
	"github.com/voxel-ai/voxel/lib/utils/go/timeutil"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestNewTimeRange(t *testing.T) {
	startTime := time.Unix(10000, 0)
	endTime := time.Unix(40000, 0)

	range1 := timeutil.NewTimeRange(startTime, endTime)
	range2 := timeutil.NewTimeRange(endTime, startTime)

	assert.Equal(t, range1.Start, startTime)
	assert.Equal(t, range1.End, endTime)

	assert.Equal(t, range2.Start, startTime)
	assert.Equal(t, range2.End, endTime)
}

func TestNewTimeRangeFromDuration(t *testing.T) {
	startTime := time.Unix(10000, 0)
	endTime := time.Unix(40000, 0)

	range1 := timeutil.NewTimeRangeFromDuration(startTime, endTime.Sub(startTime))
	range2 := timeutil.NewTimeRangeFromDuration(endTime, startTime.Sub(endTime))

	assert.Equal(t, range1.Start, startTime)
	assert.Equal(t, range1.End, endTime)

	assert.Equal(t, range2.Start, startTime)
	assert.Equal(t, range2.End, endTime)
}

func TestTimeRangeDuration(t *testing.T) {
	startTime := time.Unix(10000, 0)
	endTime := time.Unix(40000, 0)

	range1 := timeutil.NewTimeRange(startTime, endTime)

	assert.Equal(t, range1.Duration(), endTime.Sub(startTime))
}

func TestTimeRangeSeriesNormalize(t *testing.T) {
	testCases := []struct {
		name     string
		input    timeutil.TimeRangeSeries
		expected timeutil.TimeRangeSeries
	}{
		{
			name: "In-order series with no gaps",
			input: timeutil.TimeRangeSeries{
				timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(20000, 0)),
				timeutil.NewTimeRange(time.Unix(20000, 0), time.Unix(30000, 0)),
				timeutil.NewTimeRange(time.Unix(30000, 0), time.Unix(40000, 0)),
			},
			expected: timeutil.TimeRangeSeries{
				timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(40000, 0)),
			},
		},
		{
			name: "Out of order series with no gaps",
			input: timeutil.TimeRangeSeries{
				timeutil.NewTimeRange(time.Unix(30000, 0), time.Unix(40000, 0)),
				timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(20000, 0)),
				timeutil.NewTimeRange(time.Unix(20000, 0), time.Unix(30000, 0)),
			},
			expected: timeutil.TimeRangeSeries{
				timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(40000, 0)),
			},
		},
		{
			name: "In-order series with one second gaps",
			input: timeutil.TimeRangeSeries{
				timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(20000, 0)),
				timeutil.NewTimeRange(time.Unix(20001, 0), time.Unix(30000, 0)),
				timeutil.NewTimeRange(time.Unix(30001, 0), time.Unix(40000, 0)),
			},
			expected: timeutil.TimeRangeSeries{
				timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(20000, 0)),
				timeutil.NewTimeRange(time.Unix(20001, 0), time.Unix(30000, 0)),
				timeutil.NewTimeRange(time.Unix(30001, 0), time.Unix(40000, 0)),
			},
		},
		{
			name: "Out of order series with one second gaps",
			input: timeutil.TimeRangeSeries{
				timeutil.NewTimeRange(time.Unix(30001, 0), time.Unix(40000, 0)),
				timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(20000, 0)),
				timeutil.NewTimeRange(time.Unix(20001, 0), time.Unix(30000, 0)),
			},
			expected: timeutil.TimeRangeSeries{
				timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(20000, 0)),
				timeutil.NewTimeRange(time.Unix(20001, 0), time.Unix(30000, 0)),
				timeutil.NewTimeRange(time.Unix(30001, 0), time.Unix(40000, 0)),
			},
		},
		{
			name: "One dominating time range",
			input: timeutil.TimeRangeSeries{
				timeutil.NewTimeRange(time.Unix(13000, 0), time.Unix(20000, 0)),
				timeutil.NewTimeRange(time.Unix(19000, 0), time.Unix(30000, 0)),
				timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(40000, 0)),
			},
			expected: timeutil.TimeRangeSeries{
				timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(40000, 0)),
			},
		},
	}

	for _, testCase := range testCases {
		output := testCase.input.Normalize()
		assert.Equalf(t, testCase.expected, output, "test case: %v", testCase.name)
	}
}

func TestTimeRangeSeriesIsSorted(t *testing.T) {
	testCases := []struct {
		name                  string
		input                 timeutil.TimeRangeSeries
		expectedSortedByStart bool
		expectedSortedByEnd   bool
	}{
		{
			name: "In-order of start time and endTime",
			input: timeutil.TimeRangeSeries{
				timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(20000, 0)),
				timeutil.NewTimeRange(time.Unix(20000, 0), time.Unix(30000, 0)),
				timeutil.NewTimeRange(time.Unix(30000, 0), time.Unix(40000, 0)),
			},
			expectedSortedByStart: true,
			expectedSortedByEnd:   true,
		},
		{
			name: "Out of order of start time and endTime",
			input: timeutil.TimeRangeSeries{
				timeutil.NewTimeRange(time.Unix(30000, 0), time.Unix(40000, 0)),
				timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(20000, 0)),
				timeutil.NewTimeRange(time.Unix(20000, 0), time.Unix(30000, 0)),
			},
			expectedSortedByStart: false,
			expectedSortedByEnd:   false,
		},
		{
			name: "In-order of start time, out of order of endTime",
			input: timeutil.TimeRangeSeries{
				timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(40000, 0)),
				timeutil.NewTimeRange(time.Unix(20000, 0), time.Unix(30000, 0)),
				timeutil.NewTimeRange(time.Unix(30000, 0), time.Unix(50000, 0)),
			},
			expectedSortedByStart: true,
			expectedSortedByEnd:   false,
		},
		{
			name: "Out of order of start time, in-order of endTime",
			input: timeutil.TimeRangeSeries{
				timeutil.NewTimeRange(time.Unix(20000, 0), time.Unix(30000, 0)),
				timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(40000, 0)),
				timeutil.NewTimeRange(time.Unix(30000, 0), time.Unix(50000, 0)),
			},
			expectedSortedByStart: false,
			expectedSortedByEnd:   true,
		},
	}

	for _, testCase := range testCases {
		assert.Equalf(t, testCase.expectedSortedByStart, testCase.input.IsSortedByStart(), "test case: %v", testCase.name)
		assert.Equalf(t, testCase.expectedSortedByEnd, testCase.input.IsSortedByEnd(), "test case: %v", testCase.name)
	}
}

func TestTimeRangeSeriesIsContiguous(t *testing.T) {
	perfectRange := timeutil.TimeRangeSeries{
		timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(20000, 0)),
		timeutil.NewTimeRange(time.Unix(20000, 0), time.Unix(30000, 0)),
		timeutil.NewTimeRange(time.Unix(30000, 0), time.Unix(40000, 0)),
	}

	rangeWithOneSecondGaps := timeutil.TimeRangeSeries{
		timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(20000, 0)),
		timeutil.NewTimeRange(time.Unix(20001, 0), time.Unix(30000, 0)),
		timeutil.NewTimeRange(time.Unix(30001, 0), time.Unix(40000, 0)),
	}

	rangeWithSlightlyMoreThanOneSecondGap := timeutil.TimeRangeSeries{
		timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(20000, 0)),
		timeutil.NewTimeRange(time.Unix(20001, 0), time.Unix(30000, 0)),
		timeutil.NewTimeRange(time.Unix(30001, 1), time.Unix(40000, 0)),
	}

	rangeWithMinuteGaps := timeutil.TimeRangeSeries{
		timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(20000, 0)),
		timeutil.NewTimeRange(time.Unix(20060, 0), time.Unix(30000, 0)),
		timeutil.NewTimeRange(time.Unix(30060, 0), time.Unix(40000, 0)),
	}

	assert.True(t, perfectRange.IsContiguous(0))
	assert.True(t, rangeWithOneSecondGaps.IsContiguous(time.Second))
	assert.False(t, rangeWithSlightlyMoreThanOneSecondGap.IsContiguous(time.Second))
	assert.True(t, rangeWithMinuteGaps.IsContiguous(time.Minute))
}

func TestTimeRangeSeriesGetGaps(t *testing.T) {
	noGapRange := timeutil.TimeRangeSeries{
		timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(20000, 0)),
		timeutil.NewTimeRange(time.Unix(20000, 0), time.Unix(30000, 0)),
		timeutil.NewTimeRange(time.Unix(30000, 0), time.Unix(40000, 0)),
	}

	noGaps := timeutil.TimeRangeSeries{}

	rangeWithOneSecondGaps := timeutil.TimeRangeSeries{
		timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(20000, 0)),
		timeutil.NewTimeRange(time.Unix(20001, 0), time.Unix(30000, 0)),
		timeutil.NewTimeRange(time.Unix(30001, 0), time.Unix(40000, 0)),
	}

	oneSecondGaps := timeutil.TimeRangeSeries{
		timeutil.NewTimeRange(time.Unix(20000, 0), time.Unix(20001, 0)),
		timeutil.NewTimeRange(time.Unix(30000, 0), time.Unix(30001, 0)),
	}

	rangeWithLargeInconsistentGaps := timeutil.TimeRangeSeries{
		timeutil.NewTimeRange(time.Unix(10000, 0), time.Unix(20000, 0)),
		timeutil.NewTimeRange(time.Unix(25000, 0), time.Unix(30000, 0)),
		timeutil.NewTimeRange(time.Unix(409000, 1), time.Unix(509000, 0)),
	}

	inconsistentGaps := timeutil.TimeRangeSeries{
		timeutil.NewTimeRange(time.Unix(20000, 0), time.Unix(25000, 0)),
		timeutil.NewTimeRange(time.Unix(30000, 0), time.Unix(409000, 1)),
	}

	largeInconsistentGaps := timeutil.TimeRangeSeries{
		timeutil.NewTimeRange(time.Unix(30000, 0), time.Unix(409000, 1)),
	}

	assert.Equal(t, noGaps, noGapRange.GetGaps(0))
	assert.Equal(t, noGaps, rangeWithOneSecondGaps.GetGaps(time.Minute))
	assert.Equal(t, noGaps, rangeWithOneSecondGaps.GetGaps(time.Second))
	assert.Equal(t, oneSecondGaps, rangeWithOneSecondGaps.GetGaps(time.Second-time.Nanosecond))

	assert.Equal(t, inconsistentGaps, rangeWithLargeInconsistentGaps.GetGaps(time.Second))
	assert.Equal(t, inconsistentGaps, rangeWithLargeInconsistentGaps.GetGaps(5000*time.Second-time.Nanosecond))
	assert.Equal(t, largeInconsistentGaps, rangeWithLargeInconsistentGaps.GetGaps(5000*time.Second))
}
