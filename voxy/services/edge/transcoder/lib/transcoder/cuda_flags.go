package transcoder

import (
	"fmt"

	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg"
)

// SetCudaFlags sets the relevant cuda decode/scale/encode flags
func SetCudaFlags(flags *FFmpegFlags, inputCodec ffmpeg.CodecName, cfg EncoderConfig) error {
	switch inputCodec {
	case ffmpeg.CodecNameH264:
		flags.Input.Prepend("-c:v", "h264_cuvid")
	case ffmpeg.CodecNameHEVC:
		flags.Input.Prepend("-c:v", "hevc_cuvid")
	default:
		return fmt.Errorf("input video codec %v not supported", inputCodec)
	}
	flags.Input.Prepend(
		"-hwaccel", "cuda",
		"-hwaccel_output_format", "cuda",
	)

	if cfg.Scaler.Enabled {
		var scaleFilter string

		switch cfg.Scaler.Layout {
		case LayoutLandscape:
			scaleFilter = fmt.Sprintf("scale_cuda=h=%d:force_original_aspect_ratio=1", cfg.Scaler.Resolution)
		case LayoutPortrait:
			scaleFilter = fmt.Sprintf("scale_cuda=w=%d:force_original_aspect_ratio=1", cfg.Scaler.Resolution)
		default:
			return fmt.Errorf("unsupported scaler layout: %#v", cfg.Scaler.Layout)
		}

		filter := DefaultFilter
		filter.Append(scaleFilter)
		flags.Filter = filter
	}

	if cfg.Remap.Enabled {
		return fmt.Errorf("remap not implemented")
	}

	flags.Encode.Append(
		"-c:v", "hevc_nvenc",
		"-preset", "slow",
		// KVS required video frame delimiter
		"-aud", "1",
		// forces I-frames to be full refreshes as with qs
		"-forced-idr", "1",
	)

	return nil
}
