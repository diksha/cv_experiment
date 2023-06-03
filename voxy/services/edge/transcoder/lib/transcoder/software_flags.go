package transcoder

import (
	"fmt"
)

// SetSoftwareFlags sets the relevant software decode/scale/encode flags
func SetSoftwareFlags(flags *FFmpegFlags, cfg EncoderConfig) error {
	filter := DefaultFilter

	if cfg.Scaler.Enabled {
		var scaleFilter string

		switch cfg.Scaler.Layout {
		case LayoutLandscape:
			scaleFilter = fmt.Sprintf("scale=-1:%d", cfg.Scaler.Resolution)
		case LayoutPortrait:
			scaleFilter = fmt.Sprintf("scale=%d:-1", cfg.Scaler.Resolution)
		default:
			return fmt.Errorf("unsupported scaler layout: %#v", cfg.Scaler.Layout)
		}

		filter.Append(scaleFilter)
		flags.Filter = filter
	}

	if cfg.Remap.Enabled {
		flags.Input.Append(
			"-i", cfg.Remap.PGMXPath,
			"-i", cfg.Remap.PGMYPath,
		)

		filter.Append("remap")
		flags.Filter = NewRawFilter("-filter_complex", filter.String())
	}

	flags.Encode.Append(
		"-c:v", "libx264",
		"-preset", "veryfast",
	)

	return nil
}
