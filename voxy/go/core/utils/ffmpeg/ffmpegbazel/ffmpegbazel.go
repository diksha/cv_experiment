package ffmpegbazel

import (
	"fmt"
	"os/exec"

	"github.com/bazelbuild/rules_go/go/runfiles"

	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg"
)

// Find attempts to find ffmpeg by searching for it in bazel runfiles and sets their location in `github.com/voxel-ai/voxel/go/utils/ffmpeg`
// This will attempt to execute the located ffmpeg binary to ensure the found ffmpeg is usable. This package is mainly intended for use in
// tests as the only way for Go programs to use ffmpeg from bazel is if the program is executed with `bazel run` or `bazel test`
func Find() error {
	runf, err := runfiles.New()
	if err != nil {
		return fmt.Errorf("failed to find ffmpeg: %w", err)
	}

	ffmpegPath, err := runf.Rlocation("artifacts_ffmpeg_static_binary/ffmpeg")
	if err != nil {
		return fmt.Errorf("failed to find bazel ffmpeg binary: %w", err)
	}

	// trunk-ignore(semgrep/go.lang.security.audit.dangerous-exec-command.dangerous-exec-command)
	cmd := exec.Command(ffmpegPath, "-version")
	_, err = cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to find usable ffmpeg binary: %w", err)
	}

	ffprobePath, err := runf.Rlocation("artifacts_ffmpeg_static_binary/ffprobe")
	if err != nil {
		return fmt.Errorf("failed to find bazel ffprobe binary: %w", err)
	}

	// trunk-ignore(semgrep/go.lang.security.audit.dangerous-exec-command.dangerous-exec-command)
	cmd = exec.Command(ffprobePath, "-version")
	_, err = cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to find usable ffprobe binary: %w", err)
	}

	if err = ffmpeg.SetFFmpegPath(ffmpegPath); err != nil {
		return fmt.Errorf("failed to set ffmpeg path: %w", err)
	}

	if err = ffmpeg.SetFFprobePath(ffprobePath); err != nil {
		return fmt.Errorf("failed to set ffprobe path: %w", err)
	}

	return nil
}
