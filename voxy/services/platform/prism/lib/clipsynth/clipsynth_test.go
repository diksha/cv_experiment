package clipsynth_test

// trunk-ignore-all(golangci-lint/wrapcheck)

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/bazelbuild/rules_go/go/runfiles"
	"github.com/davecgh/go-spew/spew"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg"
	"github.com/voxel-ai/voxel/services/platform/prism/lib/clipsynth"
)

var fragmentFiles = []string{
	"1681431207.mkv",
	"1681431215.mkv",
	"1681431223.mkv",
	"1681431231.mkv",
}

func doTranscode(ctx context.Context, t *testing.T, inputFileNames []string, startTimestamp, endTimestamp time.Time) ([]byte, error) {
	r, err := runfiles.New()
	require.NoError(t, err)

	inputDirPath, err := r.Rlocation("artifacts_05_17_2023_clipsynth_testdata_fragments/")
	require.NoError(t, err)

	inputFilePaths := make([]string, len(inputFileNames))
	for i, filepath := range inputFileNames {
		require.Contains(t, fragmentFiles, filepath)
		inputFilePaths[i] = fmt.Sprintf("%s/%s", inputDirPath, filepath)
	}

	return clipsynth.TranscodeFiles(ctx, inputFilePaths, startTimestamp, endTimestamp)
}

func doFFProbe(ctx context.Context, t *testing.T, clipBytes []byte) (*ffmpeg.ProbeResult, error) {
	file, err := os.CreateTemp("", "clip-*.mp4")
	require.NoError(t, err)
	defer func() {
		err := os.Remove(file.Name())
		require.NoError(t, err)
	}()

	t.Log("writing clip to", file.Name())

	_, err = file.Write(clipBytes)
	require.NoError(t, err)

	return ffmpeg.ProbeFrames(ctx, ffmpeg.InputAutodetect(file.Name()))
}

func TestGenerateClip(testT *testing.T) {
	ctx := context.Background()

	testCases := []struct {
		name                 string
		inputFiles           []string
		startTime            time.Time
		endTime              time.Time
		expectedNumFrames    int
		expectedStartPTSTime string
		expectedEndPTSTime   string
		expectedDuration     string
	}{
		{
			name:                 "single fragment",
			inputFiles:           []string{fragmentFiles[0]},
			startTime:            time.UnixMilli(1681431209000),
			endTime:              time.UnixMilli(1681431212000),
			expectedNumFrames:    17,
			expectedStartPTSTime: "0.000000",
			expectedEndPTSTime:   "3.200000",
			expectedDuration:     "3.400000",
		},
		{
			name:                 "two fragments",
			inputFiles:           []string{fragmentFiles[0], fragmentFiles[1]},
			startTime:            time.UnixMilli(1681431209000),
			endTime:              time.UnixMilli(1681431219000),
			expectedNumFrames:    53,
			expectedStartPTSTime: "0.000000",
			expectedEndPTSTime:   "10.400000",
			expectedDuration:     "10.600000",
		},
		{
			name:                 "early start time",
			inputFiles:           []string{fragmentFiles[2], fragmentFiles[3]},
			startTime:            time.UnixMilli(1681431222000),
			endTime:              time.UnixMilli(1681431229000),
			expectedNumFrames:    29,
			expectedStartPTSTime: "1.600000", // still should be relative to start time, not 0
			expectedEndPTSTime:   "7.200000",
			expectedDuration:     "5.800000",
		},
		{
			name:                 "late end time",
			inputFiles:           []string{fragmentFiles[2], fragmentFiles[3]},
			startTime:            time.UnixMilli(1681431225000),
			endTime:              time.UnixMilli(1681431233000),
			expectedNumFrames:    41,
			expectedStartPTSTime: "0.200000",
			expectedEndPTSTime:   "8.200000",
			expectedDuration:     "8.200000",
		},
		{
			name:                 "mising fragment",
			inputFiles:           []string{fragmentFiles[0], fragmentFiles[2]},
			startTime:            time.UnixMilli(1681431215400),
			endTime:              time.UnixMilli(1681431223600),
			expectedNumFrames:    42,
			expectedStartPTSTime: "0.000000",
			expectedEndPTSTime:   "8.200000",
			expectedDuration:     "8.400000",
		},
	}

	for _, testCase := range testCases {
		testT.Run(testCase.name, func(t *testing.T) {
			clipBytes, err := doTranscode(ctx, t, testCase.inputFiles, testCase.startTime, testCase.endTime)
			require.NoError(t, err, "should not error on transcode")

			probeResult, err := doFFProbe(ctx, t, clipBytes)
			require.NoError(t, err, "should not error on ffprobe")

			t.Log(spew.Sdump(probeResult.Format))
			t.Log(spew.Sdump(probeResult.Streams))
			t.Log(spew.Sdump(probeResult.Frames[0]))
			t.Log(spew.Sdump(probeResult.Frames[len(probeResult.Frames)-1]))

			for _, frame := range probeResult.Frames {
				t.Log(frame.PacketPTSTime)
			}

			require.Len(t, probeResult.Frames, testCase.expectedNumFrames, "should have correct number of frames")

			assert.Equal(t, testCase.expectedStartPTSTime, probeResult.Frames[0].PacketPTSTime, "start PTS time should be correct")
			assert.Equal(t, testCase.expectedEndPTSTime, probeResult.Frames[len(probeResult.Frames)-1].PacketPTSTime, "end PTS time should be correct")

			require.Len(t, probeResult.Streams, 1, "should have one stream")
			assert.Equal(t, testCase.expectedDuration, probeResult.Streams[0].Duration, "duration should be correct")
		})
	}
}
