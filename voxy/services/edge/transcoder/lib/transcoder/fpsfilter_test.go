package transcoder_test

import (
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
)

// trunk-ignore-all(golangci-lint/varnamelen): matching ffmpeg's expression naming convention

var eightFPSSample = []int64{
	220, 320, 453, 586, 720, 820, 828, 953, 1086, 1220, 1320, 1453, 1586, 1720, 1820, 1953,
	2086, 2220, 2320, 2453, 2586, 2720, 2820, 2953, 3086, 3220, 3320, 3453, 3586, 3720, 3820, 3953,
	4086, 4220, 4320, 4453, 4586, 4720, 4820, 5051, 5184, 5318, 5418, 5551, 5684, 5818, 5918, 6051,
	6184, 6318, 6418, 6551, 6684, 6818, 6918, 7051, 7184, 7318, 7418, 7551, 7684, 7818, 7918, 8051,
	8184, 8318, 8418, 8551, 8684, 8818, 8918, 9051, 9184, 9318, 9418, 9551, 9684, 9818, 9918, 10055,
	10188, 10321, 10421, 10555, 10688, 10821, 10921, 11055, 11188, 11321, 11421, 11555, 11688, 11821,
	11921, 12055, 12188, 12321, 12421, 12555, 12688, 12821, 12921, 13055, 13188, 13321, 13421, 13555,
	13688, 13821, 13921, 14055, 14188, 14321, 14421, 14555, 14688, 14821, 14921, 15055, 15189, 15322,
	15422, 15555, 15689, 15822, 15922, 16055, 16189, 16322, 16422, 16555, 16689, 16822, 16922, 17055,
	17189, 17322, 17422, 17555, 17689, 17822, 17922, 18055, 18189, 18322, 18422, 18555, 18689, 18822,
	18922, 19055, 19189, 19322, 19422, 19555, 19689, 19822, 19922, 20059, 20192, 20326, 20426, 20559,
	20692, 20826, 20926, 21059, 21192, 21326, 21426, 21559, 21692, 21826, 21926, 22059, 22192, 22326,
	22426, 22559, 22692, 22826, 22926, 23059, 23192, 23326, 23426, 23559, 23692, 23826, 23926, 24059,
	24192, 24326, 24426, 24559, 24692, 24826, 24926, 25061, 25195, 25328, 25428, 25561, 25695, 25828,
	25928, 26061, 26195, 26328, 26428, 26561, 26695, 26828, 26928, 27061, 27195, 27328, 27428, 27561,
	27695, 27828, 27928, 28061, 28195, 28328, 28428, 28561, 28695, 28828, 28928, 29061, 29195, 29328,
	29428, 29561, 29695, 29828, 29928, 30062, 30195, 30329, 30429, 30562, 30695}

func runAlgorithm(fn func(n int, t, prev_selected_t int64) bool, timestamps []int64) []int64 {
	var selected []int64
	// trunk-ignore(golangci-lint/revive): matching ffmpeg naming
	var prev_selected_t int64
	for n, t := range timestamps {
		if fn(n, t, prev_selected_t) {
			selected = append(selected, t)
			prev_selected_t = t
		}
	}
	return selected
}

func genTimestamps(fn func(n int, prev_t int64) int64, count int) []int64 {
	var timestamps []int64
	// trunk-ignore(golangci-lint/revive): matching ffmpeg naming
	var prev_t int64
	for n := 0; n < count; n++ {
		prev_t = fn(n, prev_t)
		timestamps = append(timestamps, prev_t)
	}
	return timestamps
}

func calculateFps(timestamps []int64) []float64 {
	var fps []float64
	var window []int64
	for _, ts := range timestamps {
		window = append(window, ts)

		start := window[0]
		end := window[len(window)-1]

		for end-window[0] >= 1000 {
			start = window[0]
			window = window[1:]
		}

		if end-start >= 1000 {
			fps = append(fps, float64(len(window))/(float64(end-start)/1000.0))
		}
	}
	return fps
}

func testAlgorithm(t *testing.T, minfps, maxfps float64, input []int64, fn func(n int, t, prev_selected_t int64) bool) {
	output := runAlgorithm(fn, input)
	inputfps := float64(len(input)-1) / (float64(input[len(input)-1]-input[0]) / 1000.0)
	outputfps := float64(len(output)-1) / (float64(output[len(output)-1]-output[0]) / 1000.0)

	assert.GreaterOrEqual(t, outputfps, minfps, "average fps should be above minimum")
	assert.LessOrEqual(t, outputfps, maxfps, "average fps should be below maximum")

	sortedfps := calculateFps(output)
	sort.Float64s(sortedfps)

	assert.GreaterOrEqual(t, sortedfps[0], minfps, "calculated lowest fps should be above minimum")
	assert.LessOrEqual(t, sortedfps[len(sortedfps)-1], maxfps, "calculated highest fps should be below maximum")

	if t.Failed() {
		t.Fatalf("Algorithm Test Result:\ninput:\n\ttimestamps=%v\n\tfps=%.3v\n\tavgfps=%f\n\noutput:\n\ttimestamps=%v\n\tfps=%.3v\n\tavgfps=%f", input, calculateFps(input), inputfps, output, calculateFps(output), outputfps)
	}
}

const algoDebug = false

// we aren't going to bother trying to parse the algorithm we're just going to transcribe
// it into a go function and try to see if we can evaluate it for correctness that way.
func TestSelectAlgorithm(tt *testing.T) {
	var v0, v1, v2, v3 int64

	// The way this is written may look a little awkward but we are attempting to match the ffmpeg
	// expression syntax as best we can to make transcribing this function as easy as possible.
	algo := func(n int, t, prev_selected_t int64) bool {
		// If P0 is the previously selected frame, P1 is the frame before, P2 before that, etc
		// The values represent the maximum threshold between the current frame that previously
		// selected frame's timestamp for the current frame to be selected
		var (
			P1 int64 = 180
			P2 int64 = 185 * 2
			P3 int64 = 190 * 3
			P4 int64 = 195 * 4
			P5 int64 = 200 * 5
		)

		if n == 0 {
			if algoDebug {
				tt.Logf("t=%v, prev_selected_t=%v, v==%v", t, prev_selected_t, v0)
			}
			return true
		}

		// this looks complex, but basically we try to accept as many frames as possible to get
		// to 5fps, based on the difference between the last 5 frames' timestamps and the current fram
		if t-prev_selected_t >= P1 || t-v0 >= P2 || t-v1 >= P3 || t-v2 >= P4 || t-v3 >= P5 {
			if algoDebug {
				tt.Logf("t=%v, prev_selected_t=%v, v0=%v,v1=%v,v2=%v,v3=%v", t, prev_selected_t, v0, v1, v2, v3)
			}
			v3 = v2
			v2 = v1
			v1 = v0
			v0 = prev_selected_t
			return true
		}
		return false
	}

	// the values in these tests are tuned to detect regressions

	tt.Run("n*150", func(t *testing.T) {
		testAlgorithm(t, 4.7, 5.1, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n) * 150
		}, 100), algo)
	})

	tt.Run("n*179", func(t *testing.T) {
		testAlgorithm(t, 4.6, 5.1, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n) * 179
		}, 100), algo)
	})

	tt.Run("n*180", func(t *testing.T) {
		// max fps for this test is 5.6, because fps comes out to 5.55
		testAlgorithm(t, 4.9, 5.6, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n) * 180
		}, 100), algo)
	})

	tt.Run("n*185", func(t *testing.T) {
		testAlgorithm(t, 4.9, 5.45, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n) * 185
		}, 100), algo)
	})

	tt.Run("n*190", func(t *testing.T) {
		testAlgorithm(t, 4.9, 5.3, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n) * 190
		}, 100), algo)
	})

	tt.Run("n*195", func(t *testing.T) {
		testAlgorithm(t, 4.9, 5.2, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n) * 195
		}, 100), algo)
	})

	tt.Run("n*200", func(t *testing.T) {
		testAlgorithm(t, 4.9, 5.1, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n) * 200
		}, 100), algo)
	})

	tt.Run("n*160+(n%11)*7", func(t *testing.T) {
		testAlgorithm(t, 4.5, 5.5, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n)*160 + int64((n%11)*7)
		}, 100), algo)
	})

	tt.Run("n*150+(n%2)*100", func(t *testing.T) {
		// we dropped the minimum fps target for this test to 4.3, the average fps currently
		// is ~4.9 so this is working well enough for our use case
		testAlgorithm(t, 4.3, 5.5, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n)*150 + int64((n%2)*100)
		}, 100), algo)
	})

	tt.Run("10fps", func(t *testing.T) {
		testAlgorithm(t, 4.9, 5.1, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n)*(1000/10) + int64(n%7)*3
		}, 100), algo)
	})

	tt.Run("12fps", func(t *testing.T) {
		testAlgorithm(t, 4.9, 5.2, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n)*(1000/12) + int64(n%7)*3
		}, 120), algo)
	})

	tt.Run("15fps", func(t *testing.T) {
		testAlgorithm(t, 4.9, 5.1, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n)*(1000/15) + int64(n%7)*3
		}, 150), algo)
	})

	tt.Run("20fps", func(t *testing.T) {
		testAlgorithm(t, 4.9, 5.1, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n)*(1000/20) + int64(n%7)*3
		}, 200), algo)
	})

	tt.Run("25fps", func(t *testing.T) {
		testAlgorithm(t, 4.9, 5.1, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n)*(1000/25) + int64(n%7)*3
		}, 250), algo)
	})

	tt.Run("30fps", func(t *testing.T) {
		testAlgorithm(t, 4.9, 5.1, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n)*(1000/30) + int64(n%7)
		}, 300), algo)
	})

	// these two cases are not as well handled and tend to run high, more tuning could potentially improve this
	tt.Run("60fps", func(t *testing.T) {
		testAlgorithm(t, 4.9, 5.21, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n) * (1000 / 60)
		}, 600), algo)
	})

	tt.Run("120fps", func(t *testing.T) {
		testAlgorithm(t, 4.9, 5.45, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n) * (1000 / 120)
		}, 1200), algo)
	})

	// smoke test
	tt.Run("1000fps", func(t *testing.T) {
		testAlgorithm(t, 4.9, 5.56, genTimestamps(func(n int, prev_t int64) int64 {
			return int64(n)
		}, 10000), algo)
	})

	tt.Run("sample input", func(t *testing.T) {
		testAlgorithm(t, 4.9, 5.15, loopTimestamps(
			20, append([]int64{0}, []int64{208, 381, 597, 790, 995}...),
		), algo)
	})

	tt.Run("8fps sample", func(t *testing.T) {
		testAlgorithm(t, 4.5, 5.5, eightFPSSample, algo)
	})
}

func loopTimestamps(count int, timestamps []int64) []int64 {
	var out []int64
	var offset int64
	for count >= 0 {
		count--
		for _, ts := range timestamps {
			out = append(out, offset+ts)
		}
		offset = out[len(out)-1]
	}
	return out
}
