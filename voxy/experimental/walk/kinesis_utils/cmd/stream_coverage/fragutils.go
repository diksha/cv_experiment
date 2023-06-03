package main

import (
	"fmt"
	"sort"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideoarchivedmedia/types"
)

func getFragmentStartTime(fragment types.Fragment) time.Time {
	return aws.ToTime(fragment.ProducerTimestamp)
}

func getFragmentEndTime(fragment types.Fragment) time.Time {
	return getFragmentStartTime(fragment).Add(time.Duration(fragment.FragmentLengthInMilliseconds) * time.Millisecond)
}

// FragmentSeries wraps a list of fragments to sort and perform timesensitive checks on them
type FragmentSeries struct {
	Fragments []types.Fragment
}

// NewFragmentSeries Sorts fragments and constructs a new FragmentSeries
func NewFragmentSeries(fragments []types.Fragment) FragmentSeries {
	sort.Slice(fragments, func(i, j int) bool {
		return getFragmentStartTime(fragments[i]).Before(
			getFragmentStartTime(fragments[j]),
		)
	})
	series := FragmentSeries{Fragments: fragments}

	return series
}

// IsEmpty returns true iff there are no fragments in the series
func (series *FragmentSeries) IsEmpty() bool {
	return len(series.Fragments) == 0
}

// CheckIncludesStart returns true iff the series of fragments starts on or before the given time
// The times are truncated to the second before comparison
func (series *FragmentSeries) CheckIncludesStart(start time.Time) bool {
	if series.IsEmpty() {
		return false
	}

	start = start.Truncate(time.Second)
	seriesStart := getFragmentStartTime(series.Fragments[0]).Truncate(time.Second)

	return !seriesStart.After(start)
}

// CheckIncludesEnd returns true iff the series of fragments ends on or after the given time
// The times are truncated to the second before comparison
func (series *FragmentSeries) CheckIncludesEnd(end time.Time) bool {
	if series.IsEmpty() {
		return false
	}

	end = end.Truncate(time.Second)
	seriesEnd := getFragmentEndTime(series.Fragments[len(series.Fragments)-1]).Truncate(time.Second)

	return !seriesEnd.Before(end)
}

// ContinuityGap describes the start and end of a gap in a series of fragments
type ContinuityGap struct {
	Start, End time.Time
}

// CheckContinuity returns true iff there are no gaps between the fragments.
// It will log any gaps between fragments for visibility.
func (series *FragmentSeries) CheckContinuity(minGapSize time.Duration) (gaps []ContinuityGap) {
	if len(series.Fragments) <= 1 {
		return nil
	}

	prevEnd := getFragmentEndTime(series.Fragments[0])
	for _, frag := range series.Fragments[1:] {
		start := getFragmentStartTime(frag)
		end := getFragmentEndTime(frag)

		// Theres generally a 200ms delay between end timestamp and start timestamp
		// Adding one second of buffer to reduce noise
		if start.After(prevEnd.Add(minGapSize)) {
			gaps = append(gaps, ContinuityGap{
				Start: prevEnd,
				End:   start,
			})
		}

		prevEnd = end
	}

	return gaps
}

// TrimStart drops any fragments that end before the specified start time
func (series *FragmentSeries) TrimStart(start time.Time) {
	i := 0
	for i < len(series.Fragments) && getFragmentEndTime(series.Fragments[i]).Before(start) {
		i++
	}

	series.Fragments = series.Fragments[i:]
}

// TrimEnd drops any fragments that start after the specified end time
func (series *FragmentSeries) TrimEnd(end time.Time) {
	if len(series.Fragments) == 0 {
		return
	}

	i := len(series.Fragments) - 1
	for i >= 0 && getFragmentStartTime(series.Fragments[i]).After(end) {
		i--
	}

	series.Fragments = series.Fragments[:i+1]
}

func (series *FragmentSeries) String() string {
	if series.IsEmpty() {
		return "Empty Series"
	}
	seriesStart := getFragmentStartTime(series.Fragments[0]).UnixMilli()
	seriesEnd := getFragmentEndTime(series.Fragments[0]).UnixMilli()

	output := fmt.Sprintf("Num Fragments = %v, Range = [%v to %v]\n", len(series.Fragments), seriesStart, seriesEnd)
	for _, frag := range series.Fragments {
		output += fmt.Sprintln("\t", *frag.FragmentNumber, getFragmentStartTime(frag), "-", getFragmentEndTime(frag))
	}

	return output
}
