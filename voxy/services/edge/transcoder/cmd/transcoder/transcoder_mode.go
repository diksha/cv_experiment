package main

import (
	"context"
	"fmt"

	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg"
	"github.com/voxel-ai/voxel/services/edge/transcoder/lib/transcoder"
)

func checkQuicksync(ctx context.Context) error {
	transcode, err := ffmpeg.Start(ctx, []string{
		"-y",
		"-f", "lavfi",
		"-i", "testsrc=d=5:s=1280x720:r=12,format=yuv420p",
		"-c:v", "hevc_qsv",
		"-shortest",
		"-f", "matroska",
		"/dev/null",
	}...)
	if err != nil {
		return err
	}
	return transcode.Wait()
}

func checkCuda(ctx context.Context) error {
	transcode, err := ffmpeg.Start(ctx, []string{
		"-y",
		"-f", "lavfi",
		"-i", "testsrc=d=5:s=1280x720:r=12,format=yuv420p",
		"-c:v", "hevc_nvenc",
		"-shortest",
		"-f", "matroska",
		"/dev/null",
	}...)
	if err != nil {
		return err
	}
	return transcode.Wait()
}

func getTranscoderMode(ctx context.Context, forceSoftware bool) (transcoder.Mode, error) {
	if forceSoftware {
		return transcoder.ModeSoftware, nil
	}

	if err := checkQuicksync(ctx); err == nil {
		return transcoder.ModeQuicksync, nil
	}
	if err := checkCuda(ctx); err == nil {
		return transcoder.ModeCuda, nil
	}

	return transcoder.ModeUndefined, fmt.Errorf("no quicksync or cuda hardware found")
}
