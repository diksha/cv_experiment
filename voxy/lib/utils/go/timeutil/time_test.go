package timeutil_test

import (
	"github.com/voxel-ai/voxel/lib/utils/go/timeutil"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestRoundDown(t *testing.T) {
	almostTenThousandOne := time.Unix(10000, 999999999)
	assert.Equal(t, timeutil.RoundDown(almostTenThousandOne, time.Second), time.Unix(10000, 0))

	fiveThousandHours := time.Time{}.Add(time.Hour * 5000)
	assert.Equal(t, fiveThousandHours, timeutil.RoundDown(fiveThousandHours, 100*time.Hour))

	fiveThousandAndChangeHours := fiveThousandHours.Add(time.Minute * 31)
	t.Log(fiveThousandHours.Unix())
	t.Log(timeutil.RoundDown(fiveThousandHours, 100*time.Hour).Unix())
	t.Log(fiveThousandAndChangeHours.Unix())
	t.Log(timeutil.RoundDown(fiveThousandAndChangeHours, 100*time.Hour).Unix())
	t.Log(fiveThousandAndChangeHours.Truncate(100 * time.Hour).Unix())

	assert.Equal(t, fiveThousandHours, timeutil.RoundDown(fiveThousandAndChangeHours, 100*time.Hour))
}

func TestRoundUp(t *testing.T) {
	tenThousand := time.Time{}.Add(time.Second * 10000)
	tenThousandOne := tenThousand.Add(time.Second)
	tenThousandAndChange := tenThousand.Add(10 * time.Nanosecond)
	almostTenThousandOne := tenThousandOne.Add(-1 * time.Nanosecond)

	fiveThousandHours := time.Time{}.Add(time.Hour * 5000)
	fiveThousandAndChangeHours := fiveThousandHours.Add(time.Minute * 10)
	fiveThousandOneHours := fiveThousandHours.Add(time.Hour)

	testCases := []struct {
		input    time.Time
		duration time.Duration
		expected time.Time
	}{
		{tenThousand, time.Second, tenThousand},
		{tenThousandAndChange, time.Second, tenThousandOne},
		{almostTenThousandOne, time.Second, tenThousandOne},
		{fiveThousandHours, time.Hour, fiveThousandHours},
		{fiveThousandAndChangeHours, time.Hour, fiveThousandOneHours},
	}

	for i, testCase := range testCases {
		actual := timeutil.RoundUp(testCase.input, testCase.duration)
		assert.Equalf(t, testCase.expected, actual, "test case %d", i)
	}
}
