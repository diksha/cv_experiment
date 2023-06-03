package ffmpeg

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"time"
)

// ProbeResult holds an unmarshaled representation of the json output of ffprobe
type ProbeResult struct {
	Format  ProbeFormatResult   `json:"format"`
	Streams []ProbeStreamResult `json:"streams"`
	Frames  []ProbeFrameResult  `json:"frames"`
}

// ProbeFormatResult holds format data in a probe result
type ProbeFormatResult struct {
	Filename       string            `json:"filename"`
	NbStreams      int               `json:"nb_streams"`
	FormatName     string            `json:"format_name"`
	FormatLongName string            `json:"format_long_name"`
	StartTime      string            `json:"start_time"`
	Tags           map[string]string `json:"tags"`
}

// ProbeStreamResult holds metadata about a stream in a probe result
type ProbeStreamResult struct {
	Index         int               `json:"index"`
	CodecName     CodecName         `json:"codec_name"`
	CodecLongName string            `json:"codec_long_name"`
	CodecType     CodecType         `json:"codec_type"`
	Profile       CodecProfile      `json:"profile"`
	Width         int               `json:"width"`
	Height        int               `json:"height"`
	AvgFrameRate  string            `json:"avg_frame_rate"`
	TimeBase      string            `json:"time_base"`
	Duration      string            `json:"duration"`
	DurationTs    int               `json:"duration_ts"`
	BitRate       string            `json:"bit_rate"`
	Tags          map[string]string `json:"tags"`
}

// ProbeFrameResult holds metadata about a frame in a probe result
type ProbeFrameResult struct {
	MediaType               string `json:"media_type"`
	StreamIndex             int    `json:"stream_index"`
	KeyFrame                int    `json:"key_frame"`
	PTS                     int    `json:"pts"`
	PTSTime                 string `json:"pts_time"`
	PacketPTS               int    `json:"pkt_pts"`
	PacketPTSTime           string `json:"pkt_pts_time"`
	PacketDTS               int    `json:"pkt_dts"`
	PacketDTSTime           string `json:"pkt_dts_time"`
	BestEffortTimestamp     int    `json:"best_effort_timestamp"`
	BestEffortTimestampTime string `json:"best_effort_timestamp_time"`
	PacketDuration          int    `json:"pkt_duration"`
	PacketDurationTime      string `json:"pkt_duration_time"`
	PacketPosition          string `json:"pkt_pos"`
	PacketSize              string `json:"pkt_size"`
	Width                   int    `json:"width"`
	Height                  int    `json:"height"`
	PixelFormat             string `json:"pix_fmt"`
	SampleAspectRatio       string `json:"sample_aspect_ratio"`
	PictureType             string `json:"pict_type"`
	CodedPictureNumber      int    `json:"coded_picture_number"`
	DisplayPictureNumber    int    `json:"display_picture_number"`
	InterlacedFrame         int    `json:"interlaced_frame"`
	TopFieldFirst           int    `json:"top_field_first"`
	RepeatPict              int    `json:"repeat_pict"`
	ColorRange              string `json:"color_range"`
	ColorSpace              string `json:"color_space"`
	ColorPrimaries          string `json:"color_primaries"`
	ColorTransfer           string `json:"color_transfer"`
	ChromaLocation          string `json:"chroma_location"`
}

// CodecType represents the media type a codec can handle
type CodecType string

// Broad definitions for the types of stream codecs
const (
	CodecTypeVideo CodecType = "video"
	CodecTypeAudio CodecType = "audio"
)

// CodecName is the set of values ffprobe returns in the CodecName field
type CodecName string

// Various audio/video codec names returned by ffmpeg
const (
	CodecNameHEVC     CodecName = "hevc"
	CodecNameH264     CodecName = "h264"
	CodecNameMJPEG    CodecName = "mjpeg"
	CodecNameRawVideo CodecName = "rawvideo"
)

// CodecProfile is a set of potential values for codec profile levels
type CodecProfile string

// Values returned by ffmpeg for codec profiles
const (
	CodecProfileH264Baseline CodecProfile = "Baseline"
	CodecProfileH264Main     CodecProfile = "Main"
	CodecProfileH264High     CodecProfile = "High"
)

// ProbeFrames does the same as Probe, but also sets the -show_frames flag
func ProbeFrames(ctx context.Context, input Input) (*ProbeResult, error) {
	extraFlags := []string{
		"-show_frames",
	}
	return doProbe(ctx, input, extraFlags)
}

// ProbeDuration does the same as Probe, but also sets the -analyzeduration flag
func ProbeDuration(ctx context.Context, input Input, d time.Duration) (*ProbeResult, error) {
	return doProbe(ctx, input, []string{
		"-analyzeduration", strconv.Itoa(int(d.Microseconds())),
	})
}

// Probe performs a probe on the target input with ffprobe configured in json mode, unmarshals the output and returns it
func Probe(ctx context.Context, input Input) (*ProbeResult, error) {
	return doProbe(ctx, input, nil)
}

func doProbe(ctx context.Context, input Input, extraFlags []string) (*ProbeResult, error) {
	ffprobePath, err := FindFFprobe()
	if err != nil {
		return nil, err
	}

	flags := append([]string{
		"-v",
		"quiet",
		"-print_format",
		"json",
		"-show_format",
		"-show_streams",
	}, extraFlags...)
	flags = append(flags, input.FFprobeFlags()...)

	// trunk-ignore(semgrep/go.lang.security.audit.dangerous-exec-command.dangerous-exec-command)
	cmd := exec.CommandContext(ctx, ffprobePath, flags...)

	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("ffprobe failed: %w", err)
	}

	var res ProbeResult
	if err = json.Unmarshal(out, &res); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ffprobe result: %w", err)
	}

	return &res, nil
}
