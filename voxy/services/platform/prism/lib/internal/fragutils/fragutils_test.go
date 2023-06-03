package fragutils_test

import (
	"fmt"
	"log"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideoarchivedmedia/types"
	"github.com/stretchr/testify/assert"

	"github.com/voxel-ai/voxel/services/platform/prism/lib/internal/fragutils"
)

func getTestFragment(startTime time.Time, lengthMs int64, fragmentNumber string) types.Fragment {
	producerTimestamp := startTime
	serverTimestamp := startTime.Add(500 * time.Millisecond)
	return types.Fragment{
		FragmentLengthInMilliseconds: lengthMs,
		FragmentNumber:               &fragmentNumber,
		FragmentSizeInBytes:          122,
		ProducerTimestamp:            &producerTimestamp,
		ServerTimestamp:              &serverTimestamp,
	}
}

func getFragmentNumbers(fragments []types.Fragment) []string {
	fragmentNumbers := []string{}
	for _, fragment := range fragments {
		fragmentNumbers = append(fragmentNumbers, aws.ToString(fragment.FragmentNumber))
	}
	return fragmentNumbers
}

func createFragSeries(startTimesUnixMilli []int64, fragLengthsMs []int64) fragutils.FragmentSeries {
	frags := []types.Fragment{}
	for i, startTimeUnixMilli := range startTimesUnixMilli {
		start := time.UnixMilli(startTimeUnixMilli)
		frags = append(frags, getTestFragment(start, fragLengthsMs[i], fmt.Sprintf("%v", startTimeUnixMilli)))
	}

	return fragutils.NewFragmentSeries(frags)
}

func TestCheckContinuity(t *testing.T) {
	var series fragutils.FragmentSeries

	series = createFragSeries([]int64{1000, 2000, 3000, 4000}, []int64{1000, 1000, 1000, 1000})
	assert.Len(t, series.CheckContinuity(), 0)

	// allow one second difference
	series = createFragSeries([]int64{1000, 2000, 3000, 5000}, []int64{1000, 1000, 1000, 1000})
	assert.Len(t, series.CheckContinuity(), 0)

	// no more than one second
	series = createFragSeries([]int64{1000, 2000, 3000, 5001}, []int64{1000, 1000, 1000, 1000})
	gaps := series.CheckContinuity()
	log.Println(len(gaps))
	log.Println(gaps[0].Start.UnixMilli())
	log.Println(gaps[0].End.UnixMilli())
	assert.Equal(
		t,
		[]fragutils.ContinuityGap{{Start: time.Unix(4, 0), End: time.Unix(5, 1000000)}},
		series.CheckContinuity(),
	)
}

func TestCheckIncludesStart(t *testing.T) {
	var series fragutils.FragmentSeries

	series = createFragSeries(nil, nil)
	assert.False(t, series.CheckIncludesStart(time.UnixMilli(5000)))

	series = createFragSeries([]int64{10000, 20000, 30000, 40000}, []int64{10000, 10000, 10000, 10000})
	assert.False(t, series.CheckIncludesStart(time.UnixMilli(9999)))
	assert.True(t, series.CheckIncludesStart(time.UnixMilli(10000)))
	assert.True(t, series.CheckIncludesStart(time.UnixMilli(10001)))
}

func TestCheckIncludesEnd(t *testing.T) {
	var series fragutils.FragmentSeries

	// empty case
	series = createFragSeries(nil, nil)
	assert.False(t, series.CheckIncludesEnd(time.UnixMilli(5000)))

	series = createFragSeries([]int64{10000, 20000, 30000, 40000}, []int64{10000, 10000, 10000, 10000})
	assert.True(t, series.CheckIncludesEnd(time.UnixMilli(40999)))
	assert.True(t, series.CheckIncludesEnd(time.UnixMilli(50000)))
	assert.True(t, series.CheckIncludesEnd(time.UnixMilli(50999)))
	assert.False(t, series.CheckIncludesEnd(time.UnixMilli(60000)))
}

func TestTrimStart(t *testing.T) {
	starts := []int64{10000, 11000, 12000, 13000, 14000, 15000}
	lengths := []int64{1000, 1000, 1000, 1000, 1000, 1000}

	var series fragutils.FragmentSeries

	// test no fatal on empty
	series = fragutils.NewFragmentSeries(nil)
	series.TrimStart(time.UnixMilli(12000))

	trimValToExpectation := map[int64][]string{
		100:    {"10000", "11000", "12000", "13000", "14000", "15000"},
		900000: {},
		12500:  {"12000", "13000", "14000", "15000"},
		12999:  {"12000", "13000", "14000", "15000"},
		13000:  {"12000", "13000", "14000", "15000"},
		13001:  {"13000", "14000", "15000"},
	}

	for trim, expecation := range trimValToExpectation {
		series = createFragSeries(starts, lengths)
		series.TrimStart(time.UnixMilli(trim))
		assert.Equal(t, getFragmentNumbers(series.Fragments), expecation)
	}
}

func TestTrimEnd(t *testing.T) {
	starts := []int64{10000, 11000, 12000, 13000, 14000, 15000}
	lengths := []int64{1000, 1000, 1000, 1000, 1000, 1000}

	var series fragutils.FragmentSeries

	// Test for fatals with empty
	series = fragutils.NewFragmentSeries(nil)
	series.TrimEnd(time.UnixMilli(12000))

	// Test for fatals with one
	series = createFragSeries([]int64{10000}, []int64{10000})
	series.TrimEnd(time.UnixMilli(12000))

	trimValToExpectation := map[int64][]string{
		15001: {"10000", "11000", "12000", "13000", "14000", "15000"},
		1000:  {},
		12500: {"10000", "11000", "12000"},
		12999: {"10000", "11000", "12000"},
		13000: {"10000", "11000", "12000", "13000"},
		13001: {"10000", "11000", "12000", "13000"},
	}

	for trim, expecation := range trimValToExpectation {
		series = createFragSeries(starts, lengths)
		series.TrimEnd(time.UnixMilli(trim))
		assert.Equal(t, getFragmentNumbers(series.Fragments), expecation)
	}
}
