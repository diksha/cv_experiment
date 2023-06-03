package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/aws/aws-lambda-go/lambda"

	"github.com/bazelbuild/rules_go/go/runfiles"

	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg"
)

var debug = flag.Bool("debug", false, "debug mode skips lambda startup")

var mediaPath string

func init() {
	r, err := runfiles.New()
	if err != nil {
		panic(err)
	}

	mediaPath, err = r.Rlocation("artifacts_office_cam_720p_mkv/office_cam_720p.mkv")
	if err != nil {
		panic(err)
	}

	ffmpegPath, err := r.Rlocation("voxel/experimental/jorge/lambda-transcoder/ffmpeg")
	if err != nil {
		panic(err)
	}

	err = ffmpeg.SetFFmpegPath(ffmpegPath)
	if err != nil {
		panic(err)
	}
}

type Result struct {
	DurationSeconds float64
}

func handler(ctx context.Context) (*Result, error) {
	cmd, err := ffmpeg.New(ctx, "-i", mediaPath, "-c:v", "libx264", "-y", "/tmp/clip.mp4")
	cmd.LogLevel = "info"
	if err != nil {
		return nil, fmt.Errorf("transcode init error: %w", err)
	}

	cmd.Stderr = os.Stderr

	start := time.Now()
	err = cmd.Run()
	if err != nil {
		return nil, fmt.Errorf("transcode failure: %w", err)
	}
	duration := time.Since(start)

	log.Printf("Transcode took %v seconds", duration)
	return &Result{
		DurationSeconds: duration.Seconds(),
	}, nil
}

func main() {
	log.SetFlags(0)
	flag.Parse()

	if *debug {
		log.Fatal(handler(context.Background()))
	}

	lambda.Start(handler)
}
