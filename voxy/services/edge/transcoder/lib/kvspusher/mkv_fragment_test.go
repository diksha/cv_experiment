package kvspusher_test

import (
	"bytes"
	"context"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg"
	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg/ffmpegbazel"
	"github.com/voxel-ai/voxel/services/edge/transcoder/lib/kvspusher"

	"github.com/bazelbuild/rules_go/go/tools/bazel"
)

func init() {
	tryFFmpegBazel()
}

// attempts to load ffmpeg from the bazel env if we are running in bazel
func tryFFmpegBazel() {
	if _, err := bazel.RunfilesPath(); err != nil {
		return
	}

	// we panic if we can confirm we are running in bazel but fail to find ffmpeg
	if err := ffmpegbazel.Find(); err != nil {
		panic(err)
	}
}

func onlyRunsWithBazel(t *testing.T) {
	if _, err := bazel.RunfilesPath(); err != nil {
		t.Skip("This test can only run under `bazel run` or `bazel test`")
	}
}

/*
	EBML Dump of H264 Test Fragment

HandleMasterBegin(EBML, {Offset:5 Size:35 Level:0})

	HandleInteger(EBMLVersion, 1, {Offset:8 Size:1 Level:1})
	HandleInteger(EBMLReadVersion, 1, {Offset:12 Size:1 Level:1})
	HandleInteger(EBMLMaxIDLength, 4, {Offset:16 Size:1 Level:1})
	HandleInteger(EBMLMaxSizeLength, 8, {Offset:20 Size:1 Level:1})
	HandleString(DocType, "matroska", {Offset:24 Size:8 Level:1})
	HandleInteger(DocTypeVersion, 4, {Offset:35 Size:1 Level:1})
	HandleInteger(DocTypeReadVersion, 2, {Offset:39 Size:1 Level:1})

HandleMasterEnd(EBML, {Offset:5 Size:35 Level:0})
HandleMasterBegin(Segment, {Offset:47 Size:104228 Level:0})

	HandleMasterBegin(Info, {Offset:52 Size:64 Level:1})
	        HandleInteger(TimecodeScale, 1000000, {Offset:56 Size:3 Level:2})
	        HandleBinary(SegmentUID, bytes len(16), {Offset:62 Size:16 Level:2})
	        HandleString(SegmentFilename, "", {Offset:81 Size:0 Level:2})
	        HandleString(Title, "", {Offset:84 Size:0 Level:2})
	        HandleString(MuxingApp, "Lavf58.29.100", {Offset:87 Size:13 Level:2})
	        HandleString(WritingApp, "Lavf58.29.100", {Offset:103 Size:13 Level:2})
	HandleMasterEnd(Info, {Offset:52 Size:64 Level:1})
	HandleMasterBegin(Tracks, {Offset:121 Size:95 Level:1})
	        HandleMasterBegin(TrackEntry, {Offset:123 Size:93 Level:2})
	                HandleInteger(TrackNumber, 1, {Offset:125 Size:1 Level:3})
	                HandleInteger(TrackUID, 1, {Offset:129 Size:1 Level:3})
	                HandleInteger(FlagLacing, 0, {Offset:132 Size:1 Level:3})
	                HandleString(CodecID, "V_MPEG4/ISO/AVC", {Offset:135 Size:15 Level:3})
	                HandleInteger(TrackType, 1, {Offset:152 Size:1 Level:3})
	                HandleInteger(DefaultDuration, 200000000, {Offset:157 Size:4 Level:3})
	                HandleMasterBegin(Video, {Offset:163 Size:8 Level:3})
	                        HandleInteger(PixelWidth, 1280, {Offset:165 Size:2 Level:4})
	                        HandleInteger(PixelHeight, 720, {Offset:169 Size:2 Level:4})
	                HandleMasterEnd(Video, {Offset:163 Size:8 Level:3})
	                HandleBinary(CodecPrivate, bytes len(42), {Offset:174 Size:42 Level:3})
	        HandleMasterEnd(TrackEntry, {Offset:123 Size:93 Level:2})
	HandleMasterEnd(Tracks, {Offset:121 Size:95 Level:1})
	HandleMasterBegin(Cluster, {Offset:223 Size:25478 Level:1})
	        HandleInteger(Timecode, 1659386574200, {Offset:225 Size:6 Level:2})
	        HandleBinary(SimpleBlock, bytes len(10435), {Offset:234 Size:10435 Level:2})
	        TrackNumber=1 Timecode=0
	        HandleBinary(SimpleBlock, bytes len(2729), {Offset:10672 Size:2729 Level:2})
	        TrackNumber=1 Timecode=200
	        HandleBinary(SimpleBlock, bytes len(3203), {Offset:13404 Size:3203 Level:2})
	        TrackNumber=1 Timecode=400
	        HandleBinary(SimpleBlock, bytes len(3191), {Offset:16610 Size:3191 Level:2})
	        TrackNumber=1 Timecode=600
	        HandleBinary(SimpleBlock, bytes len(3177), {Offset:19804 Size:3177 Level:2})
	        TrackNumber=1 Timecode=800
	        HandleBinary(SimpleBlock, bytes len(2717), {Offset:22984 Size:2717 Level:2})
	        TrackNumber=1 Timecode=1000
	HandleMasterEnd(Cluster, {Offset:223 Size:25478 Level:1})
	HandleMasterBegin(Cluster, {Offset:25707 Size:14031 Level:1})
	        HandleInteger(Timecode, 1659386575400, {Offset:25709 Size:6 Level:2})
	        HandleBinary(SimpleBlock, bytes len(2382), {Offset:25718 Size:2382 Level:2})
	        TrackNumber=1 Timecode=0
	        HandleBinary(SimpleBlock, bytes len(2470), {Offset:28103 Size:2470 Level:2})
	        TrackNumber=1 Timecode=200
	        HandleBinary(SimpleBlock, bytes len(2738), {Offset:30576 Size:2738 Level:2})
	        TrackNumber=1 Timecode=400
	        HandleBinary(SimpleBlock, bytes len(2721), {Offset:33317 Size:2721 Level:2})
	        TrackNumber=1 Timecode=600
	        HandleBinary(SimpleBlock, bytes len(1818), {Offset:36041 Size:1818 Level:2})
	        TrackNumber=1 Timecode=800
	        HandleBinary(SimpleBlock, bytes len(1876), {Offset:37862 Size:1876 Level:2})
	        TrackNumber=1 Timecode=1000
	HandleMasterEnd(Cluster, {Offset:25707 Size:14031 Level:1})
	HandleMasterBegin(Cluster, {Offset:39744 Size:13789 Level:1})
	        HandleInteger(Timecode, 1659386576600, {Offset:39746 Size:6 Level:2})
	        HandleBinary(SimpleBlock, bytes len(2009), {Offset:39755 Size:2009 Level:2})
	        TrackNumber=1 Timecode=0
	        HandleBinary(SimpleBlock, bytes len(2440), {Offset:41767 Size:2440 Level:2})
	        TrackNumber=1 Timecode=200
	        HandleBinary(SimpleBlock, bytes len(2668), {Offset:44210 Size:2668 Level:2})
	        TrackNumber=1 Timecode=400
	        HandleBinary(SimpleBlock, bytes len(1976), {Offset:46881 Size:1976 Level:2})
	        TrackNumber=1 Timecode=600
	        HandleBinary(SimpleBlock, bytes len(2227), {Offset:48860 Size:2227 Level:2})
	        TrackNumber=1 Timecode=800
	        HandleBinary(SimpleBlock, bytes len(2443), {Offset:51090 Size:2443 Level:2})
	        TrackNumber=1 Timecode=1000
	HandleMasterEnd(Cluster, {Offset:39744 Size:13789 Level:1})
	HandleMasterBegin(Cluster, {Offset:53539 Size:13893 Level:1})
	        HandleInteger(Timecode, 1659386577800, {Offset:53541 Size:6 Level:2})
	        HandleBinary(SimpleBlock, bytes len(2403), {Offset:53550 Size:2403 Level:2})
	        TrackNumber=1 Timecode=0
	        HandleBinary(SimpleBlock, bytes len(2785), {Offset:55956 Size:2785 Level:2})
	        TrackNumber=1 Timecode=200
	        HandleBinary(SimpleBlock, bytes len(2000), {Offset:58744 Size:2000 Level:2})
	        TrackNumber=1 Timecode=400
	        HandleBinary(SimpleBlock, bytes len(2058), {Offset:60747 Size:2058 Level:2})
	        TrackNumber=1 Timecode=600
	        HandleBinary(SimpleBlock, bytes len(2077), {Offset:62808 Size:2077 Level:2})
	        TrackNumber=1 Timecode=800
	        HandleBinary(SimpleBlock, bytes len(2544), {Offset:64888 Size:2544 Level:2})
	        TrackNumber=1 Timecode=1000
	HandleMasterEnd(Cluster, {Offset:53539 Size:13893 Level:1})
	HandleMasterBegin(Cluster, {Offset:67438 Size:13977 Level:1})
	        HandleInteger(Timecode, 1659386579000, {Offset:67440 Size:6 Level:2})
	        HandleBinary(SimpleBlock, bytes len(2691), {Offset:67449 Size:2691 Level:2})
	        TrackNumber=1 Timecode=0
	        HandleBinary(SimpleBlock, bytes len(2017), {Offset:70143 Size:2017 Level:2})
	        TrackNumber=1 Timecode=200
	        HandleBinary(SimpleBlock, bytes len(2091), {Offset:72163 Size:2091 Level:2})
	        TrackNumber=1 Timecode=400
	        HandleBinary(SimpleBlock, bytes len(2019), {Offset:74257 Size:2019 Level:2})
	        TrackNumber=1 Timecode=600
	        HandleBinary(SimpleBlock, bytes len(2637), {Offset:76279 Size:2637 Level:2})
	        TrackNumber=1 Timecode=800
	        HandleBinary(SimpleBlock, bytes len(2496), {Offset:78919 Size:2496 Level:2})
	        TrackNumber=1 Timecode=1000
	HandleMasterEnd(Cluster, {Offset:67438 Size:13977 Level:1})
	HandleMasterBegin(Cluster, {Offset:81421 Size:13119 Level:1})
	        HandleInteger(Timecode, 1659386580200, {Offset:81423 Size:6 Level:2})
	        HandleBinary(SimpleBlock, bytes len(1923), {Offset:81432 Size:1923 Level:2})
	        TrackNumber=1 Timecode=0
	        HandleBinary(SimpleBlock, bytes len(1855), {Offset:83358 Size:1855 Level:2})
	        TrackNumber=1 Timecode=200
	        HandleBinary(SimpleBlock, bytes len(2077), {Offset:85216 Size:2077 Level:2})
	        TrackNumber=1 Timecode=400
	        HandleBinary(SimpleBlock, bytes len(2507), {Offset:87296 Size:2507 Level:2})
	        TrackNumber=1 Timecode=600
	        HandleBinary(SimpleBlock, bytes len(2490), {Offset:89806 Size:2490 Level:2})
	        TrackNumber=1 Timecode=800
	        HandleBinary(SimpleBlock, bytes len(2241), {Offset:92299 Size:2241 Level:2})
	        TrackNumber=1 Timecode=1000
	HandleMasterEnd(Cluster, {Offset:81421 Size:13119 Level:1})
	HandleMasterBegin(Cluster, {Offset:94546 Size:9729 Level:1})
	        HandleInteger(Timecode, 1659386581400, {Offset:94548 Size:6 Level:2})
	        HandleBinary(SimpleBlock, bytes len(2276), {Offset:94557 Size:2276 Level:2})
	        TrackNumber=1 Timecode=0
	        HandleBinary(SimpleBlock, bytes len(2224), {Offset:96836 Size:2224 Level:2})
	        TrackNumber=1 Timecode=200
	        HandleBinary(SimpleBlock, bytes len(2563), {Offset:99063 Size:2563 Level:2})
	        TrackNumber=1 Timecode=400
	        HandleBinary(SimpleBlock, bytes len(2646), {Offset:101629 Size:2646 Level:2})
	        TrackNumber=1 Timecode=600
	HandleMasterEnd(Cluster, {Offset:94546 Size:9729 Level:1})

HandleMasterEnd(Segment, {Offset:47 Size:104228 Level:0})
*/
func mustLoadH264TestFragment(t *testing.T) *kvspusher.Fragment {
	chunkData := mustGetRunfile(t, "7_cluster_test_chunk.mkv")
	frag, err := kvspusher.ReadFragment(bytes.NewReader(chunkData))
	require.NoError(t, err, "does not error")
	require.NotNil(t, frag, "returns a fragment")
	return frag
}

func mustLoadHEVCTestFragment(t *testing.T) *kvspusher.Fragment {
	chunkData := mustGetRunfile(t, "2_cluster_hevc_testfile.mkv")
	frag, err := kvspusher.ReadFragment(bytes.NewReader(chunkData))
	require.NoError(t, err)
	return frag
}

func mustGetRunfilePath(t *testing.T, filename string) string {
	filepath, err := bazel.Runfile(filename)
	if err != nil {
		runfiles, _ := bazel.ListRunfiles()
		t.Error(spew.Sdump(runfiles))
		t.Fatalf("failed to find runfile %q: %v", filename, err)
	}
	return filepath
}

func mustGetRunfile(t *testing.T, filename string) []byte {
	filepath := mustGetRunfilePath(t, filename)
	data, err := ioutil.ReadFile(filepath)
	if err != nil {
		t.Fatalf("failed to load testdata %q: %v", filename, err)
	}
	return data
}

func mustProbeFragment(t *testing.T, frag *kvspusher.Fragment) *ffmpeg.ProbeResult {
	var buf bytes.Buffer
	err := kvspusher.WriteFragment(&buf, frag)
	require.NoError(t, err, "fragment marshals correctly")

	tmpdir, err := bazel.NewTmpDir("kvspusher_test")
	require.NoError(t, err, "creates a tempdir")
	defer func() { _ = os.RemoveAll(tmpdir) }()

	testFilename := filepath.Join(tmpdir, "test.mkv")
	err = ioutil.WriteFile(testFilename, buf.Bytes(), 0666)
	require.NoError(t, err, "writes a test file")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	res, err := ffmpeg.Probe(ctx, ffmpeg.InputAutodetect(testFilename))
	require.NoError(t, err, "testfile probe does not fail")

	return res
}

func mustValidateFragment(t *testing.T, frag *kvspusher.Fragment) {
	var buf bytes.Buffer
	err := kvspusher.WriteFragment(&buf, frag)
	require.NoError(t, err, "fragment marshals correctly")

	tmpdir, err := bazel.NewTmpDir("kvspusher_test")
	require.NoError(t, err, "creates a tempdir")
	defer func() { _ = os.RemoveAll(tmpdir) }()

	testFilename := filepath.Join(tmpdir, "test.mkv")
	err = ioutil.WriteFile(testFilename, buf.Bytes(), 0666)
	require.NoError(t, err, "writes a test file")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var stderr bytes.Buffer
	cmd, err := ffmpeg.New(ctx, "-xerror", "-i", testFilename, "-f", "null", "-")
	cmd.LogLevel = "debug"
	cmd.Stderr = &stderr
	require.NoError(t, err, "create ffmpeg cmd without error")
	assert.NoError(t, cmd.Run(), "run ffmpeg without error")
	t.Log(stderr.String())
}

func TestLoadFragment(t *testing.T) {
	onlyRunsWithBazel(t)

	frag := mustLoadH264TestFragment(t)

	assert.Len(t, frag.Segment.Cluster, 7, "cluster count is correct")
	assert.Len(t, frag.Segment.Tracks.TrackEntry, 1, "track count is correct")
	assert.EqualValues(t, 1, frag.Segment.Tracks.TrackEntry[0].TrackNumber, "track number is correct")
	assert.Equal(t, "V_MPEG4/ISO/AVC", frag.Segment.Tracks.TrackEntry[0].CodecID, "codec id is correct")
}

func TestWriteFragment(t *testing.T) {
	onlyRunsWithBazel(t)

	inFrag := mustLoadH264TestFragment(t)

	var buf bytes.Buffer
	err := kvspusher.WriteFragment(&buf, inFrag)
	require.NoError(t, err, "writes output fragment")

	outFrag, err := kvspusher.ReadFragment(bytes.NewReader(buf.Bytes()))
	require.NoError(t, err, "reads output fragment")

	assert.Len(t, outFrag.Segment.Cluster, 7, "output cluster count is correct")

	res := mustProbeFragment(t, outFrag)

	assert.EqualValues(t, 1, res.Format.NbStreams, "stream count is correct")
	assert.EqualValues(t, 1280, res.Streams[0].Width, "width is correct")
	assert.EqualValues(t, 720, res.Streams[0].Height, "height is correct")
}

func TestMergeClusters(t *testing.T) {
	onlyRunsWithBazel(t)

	frag := mustLoadH264TestFragment(t)

	blockCount := 0
	for _, c := range frag.Segment.Cluster {
		blockCount += len(c.SimpleBlock)
	}

	merged := kvspusher.MergeClusters(frag.Segment.Cluster)

	assert.EqualValues(t, blockCount, len(merged.SimpleBlock), "block count matches after merging")
}

func TestSimplifyFragmentH264(t *testing.T) {
	inFrag := mustLoadH264TestFragment(t)
	outFrag := kvspusher.SimplifyFragment(inFrag)

	probeRes := mustProbeFragment(t, outFrag)
	assert.EqualValues(t, 1, probeRes.Format.NbStreams, "should have 1 stream")

	mustValidateFragment(t, outFrag)
}

func TestSimplifyFragmentHEVC(t *testing.T) {
	inFrag := mustLoadHEVCTestFragment(t)
	outFrag := kvspusher.SimplifyFragment(inFrag)

	probeRes := mustProbeFragment(t, outFrag)
	assert.EqualValues(t, 1, probeRes.Format.NbStreams, "should have 1 stream")

	mustValidateFragment(t, outFrag)
}

func TestGetAllTimestamps(t *testing.T) {
	frag := mustLoadH264TestFragment(t)
	timestamps := frag.GetAllTimestamps()

	require.Len(t, timestamps, 40)
	assert.EqualValues(t, []time.Duration{
		1659386574200000000, 1659386574400000000, 1659386574600000000, 1659386574800000000, 1659386575000000000, 1659386575200000000, 1659386575400000000, 1659386575600000000, 1659386575800000000, 1659386576000000000, 1659386576200000000, 1659386576400000000, 1659386576600000000, 1659386576800000000, 1659386577000000000, 1659386577200000000, 1659386577400000000, 1659386577600000000, 1659386577800000000, 1659386578000000000, 1659386578200000000, 1659386578400000000, 1659386578600000000, 1659386578800000000, 1659386579000000000, 1659386579200000000, 1659386579400000000, 1659386579600000000, 1659386579800000000, 1659386580000000000, 1659386580200000000, 1659386580400000000, 1659386580600000000, 1659386580800000000, 1659386581000000000, 1659386581200000000, 1659386581400000000, 1659386581600000000, 1659386581800000000, 1659386582000000000,
	}, timestamps)
}

func TestGetMinTimestamp(t *testing.T) {
	frag := mustLoadH264TestFragment(t)
	timestamp, err := frag.MinTimestamp()

	require.NoError(t, err)
	assert.Equal(t, 1659386574200000000*time.Nanosecond, timestamp)
}

func TestGetMaxTimestamp(t *testing.T) {
	frag := mustLoadH264TestFragment(t)
	timestamp, err := frag.MaxTimestamp()

	require.NoError(t, err)
	assert.Equal(t, 1659386582000000000*time.Nanosecond, timestamp)
}

func TestDuration(t *testing.T) {
	frag := mustLoadH264TestFragment(t)
	duration, err := frag.Duration()

	require.NoError(t, err)
	assert.Equal(t, 7800*time.Millisecond, duration)
}
