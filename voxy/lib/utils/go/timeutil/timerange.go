package timeutil

import (
	"sort"
	"time"
)

// TimeRange represents a continuous range of time
type TimeRange struct {
	Start time.Time
	End   time.Time
}

// NewTimeRange creates a new TimeRange
func NewTimeRange(t1 time.Time, t2 time.Time) TimeRange {
	start := t1
	end := t2

	if t2.Before(t1) {
		start = t2
		end = t1
	}

	return TimeRange{
		Start: start,
		End:   end,
	}
}

// NewTimeRangeFromDuration creates a new TimeRange from a start time and a duration
func NewTimeRangeFromDuration(t time.Time, duration time.Duration) TimeRange {
	if duration < 0 {
		return NewTimeRange(t.Add(duration), t)
	}

	return NewTimeRange(t, t.Add(duration))
}

// Duration returns the duration of the TimeRange
func (tr TimeRange) Duration() time.Duration {
	return tr.End.Sub(tr.Start)
}

// TimeRangeSeries is a list of TimeRanges
type TimeRangeSeries []TimeRange

// Normalize will take a given time series and sort it by start time, and merge any overlapping time ranges.
func (series TimeRangeSeries) Normalize() TimeRangeSeries {
	series.SortByStart()

	for i := 0; i < len(series)-1; i++ {
		if !series[i].End.Before(series[i+1].Start) {
			if series[i+1].End.After(series[i].End) {
				series[i].End = series[i+1].End
			}

			series = append(series[:i+1], series[i+2:]...)
			i--
		}
	}

	return series
}

// SortByStart sorts the time ranges by their start time
func (series TimeRangeSeries) SortByStart() {
	sort.Slice(series, func(i, j int) bool {
		return series[i].Start.Before(series[j].Start)
	})
}

// IsSortedByStart returns true if the time ranges are sorted by their start time
func (series TimeRangeSeries) IsSortedByStart() bool {
	return sort.SliceIsSorted(series, func(i, j int) bool {
		return series[i].Start.Before(series[j].Start)
	})
}

// IsSortedByEnd returns true if the time ranges are sorted by their end time
func (series TimeRangeSeries) IsSortedByEnd() bool {
	return sort.SliceIsSorted(series, func(i, j int) bool {
		return series[i].End.Before(series[j].End)
	})
}

// IsContiguous returns true if the given time ranges are continuous.
// A time range series is contiguous if the end of one range is no more than maxAllowableGap before the start of the next range for each range in the series.
// Series should be normalized.
func (series TimeRangeSeries) IsContiguous(maxAllowableGap time.Duration) bool {
	return len(series.GetGaps(maxAllowableGap)) == 0
}

// GetGaps returns a TimeRangeSeries containing the gaps between the given time ranges.
// Series should be normalized.
func (series TimeRangeSeries) GetGaps(maxAllowableGap time.Duration) TimeRangeSeries {
	gaps := TimeRangeSeries{}

	for i := 0; i < len(series)-1; i++ {
		if series[i].End.Add(maxAllowableGap).Before(series[i+1].Start) {
			gaps = append(gaps, NewTimeRange(series[i].End, series[i+1].Start))
		}
	}

	return gaps
}
