package ffmpeg

import (
	"fmt"
	"os/exec"
)

var (
	ffprobeBinPath string
	ffmpegBinPath  string
)

func findBin(name string) (string, error) {
	p, err := exec.LookPath(name)
	if err != nil {
		return "", fmt.Errorf("failed to find %s: %w", name, err)
	}
	return p, nil
}

// SetFFprobePath should only be called during program setup and sets
// the path to the ffprobe binary
func SetFFprobePath(p string) error {
	res, err := exec.LookPath(p)
	if err != nil {
		return fmt.Errorf("invalid ffprobe path %q: %w", p, err)
	}
	ffprobeBinPath = res
	return nil
}

// SetFFmpegPath should only be called during program setup and sets
// the path to the ffmpeg binary
func SetFFmpegPath(p string) error {
	res, err := exec.LookPath(p)
	if err != nil {
		return fmt.Errorf("invalid ffmpeg path %q: %w", p, err)
	}
	ffmpegBinPath = res
	return nil
}

// FindFFprobe attempts to find the system ffprobe and returns its path
func FindFFprobe() (string, error) {
	if ffprobeBinPath != "" {
		return ffprobeBinPath, nil
	}
	return findBin("ffprobe")
}

// FindFFmpeg attempts to find the system ffmpeg and returns its path
func FindFFmpeg() (string, error) {
	if ffmpegBinPath != "" {
		return ffmpegBinPath, nil
	}
	return findBin("ffmpeg")
}

// MustFindFFprobe attempts to find ffprobe and panics if it fails
func MustFindFFprobe() string {
	p, err := FindFFprobe()
	if err != nil {
		panic(err)
	}
	return p
}

// MustFindFFmpeg attempts to find ffmpeg and panics if it fails
func MustFindFFmpeg() string {
	p, err := FindFFmpeg()
	if err != nil {
		panic(err)
	}
	return p
}
