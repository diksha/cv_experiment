package transcoder

import (
	"fmt"
	"strings"

	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg"
)

// SetQuicksyncFlags sets the relevant quicksync decode/scale/encode flags
func SetQuicksyncFlags(flags *FFmpegFlags, inputStream ffmpeg.ProbeStreamResult, cfg EncoderConfig) error {
	switch inputStream.CodecName {
	case ffmpeg.CodecNameH264:
		flags.Input.Prepend("-c:v", "h264_qsv")
	case ffmpeg.CodecNameHEVC:
		flags.Input.Prepend("-c:v", "hevc_qsv")
	case ffmpeg.CodecNameMJPEG:
		flags.Input.Prepend("-c:v", "mjpeg_qsv")
	default:
		return fmt.Errorf("input video codec %v not supported", inputStream.CodecName)
	}

	// set decoder flags
	flags.Input.Prepend(
		"-hwaccel", "qsv",
		"-hwaccel_output_format", "qsv",
	)

	var scaleFilter string
	// configure but don't set the scale filter yet because we might have to remap first
	if cfg.Scaler.Enabled {
		switch cfg.Scaler.Layout {
		case LayoutLandscape:
			scaleFilter = fmt.Sprintf("vpp_qsv=w=in_w*h/in_h:h=%d", cfg.Scaler.Resolution)
		case LayoutPortrait:
			scaleFilter = fmt.Sprintf("vpp_qsv=w=%d:h=in_h*w/in_w", cfg.Scaler.Resolution)
		default:
			return fmt.Errorf("unsupported scaler layout %#v", cfg.Scaler.Layout)
		}
	}

	if cfg.Remap.Enabled {
		// these lines set up the qsv->va mapping which is necessary to map qsv into opencl memory
		flags.Input.Prepend(
			"-init_hw_device", "vaapi=va,driver=iHD",
			"-init_hw_device", "qsv=qs@va",
			"-filter_hw_device", "qs",
		)

		flags.Input.Append(
			"-i", cfg.Remap.PGMXPath,
			"-i", cfg.Remap.PGMYPath,
		)

		// we could attach other extra filters here in the future
		// but for now it's just the scale filter
		extraFilter := scaleFilter

		// add a comma before extrafilter if we have one to make the filter line legible
		if len(extraFilter) > 0 {
			extraFilter = "," + extraFilter
		}

		// the filter configuration below does a format conversion in system memory to better support the
		// remap_opencl filter as that filter does not support the nv12 pixel format. If/when we are able
		// to get a remap_opencl filter that supports nv12 we should revert to the original hwmap approach
		flags.Filter = NewRawFilter("-filter_complex", strings.Join([]string{
			// set up the base filter as an input and memory map qsv memory into opencl
			fmt.Sprintf("[0:0]%s,hwdownload,format=nv12,format=bgra,hwupload=derive_device=opencl[input]", DefaultFilter.String()),
			// upload the remap_x.pgm file into gpu memory
			"[1:0]hwupload=derive_device=opencl[pgmx]",
			// upload the remap_y.pgm file into gpu memory
			"[2:0]hwupload=derive_device=opencl[pgmy]",
			// map inputs to the remap_opencl filter and then hwmap back to qsv
			"[input][pgmx][pgmy]remap_opencl,hwdownload,format=bgra,format=nv12,hwupload=derive_device=qsv:extra_hw_frames=16" + extraFilter,
		}, ";"))

		// flags.Filter = NewRawFilter("-filter_complex", strings.Join([]string{
		// 	// set up the base filter as an input and memory map qsv memory into opencl
		// 	fmt.Sprintf("[0:0]%s,hwmap=derive_device=opencl[input]", DefaultFilter.String()),
		// 	// upload the remap_x.pgm file into gpu memory
		// 	"[1:0]hwupload=derive_device=opencl[pgmx]",
		// 	// upload the remap_y.pgm file into gpu memory
		// 	"[2:0]hwupload=derive_device=opencl[pgmy]",
		// 	// map inputs to the remap_opencl filter and then hwmap back to qsv
		// 	"[input][pgmx][pgmy]remap_opencl,hwmap=reverse=1:derive_device=qsv:extra_hw_frames=16" + extraFilter,
		// }, ";"))
	} else if scaleFilter != "" {
		// if remap isn't set and we have a scale filter, make sure to add it
		filter := DefaultFilter
		filter.Append(scaleFilter)
		flags.Filter = filter
	}

	flags.Encode.Append(
		// turn on the hardware hevc encoder
		"-c:v", "hevc_qsv",
		// aud = access use delimiter = start code for video frames required by kvs
		"-aud", "1",
		// forces I-frames to be full refreshes, without this our keyframe
		// logic does not work
		"-forced_idr", "1",
	)

	return nil
}
