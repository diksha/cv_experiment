package fish2persp

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/bazelbuild/rules_go/go/tools/bazel"
	"github.com/rs/zerolog/log"

	edgeconfigpb "github.com/voxel-ai/voxel/protos/edge/edgeconfig/v1"
)

var fish2perspLocations = []string{
	"fish2persp",
	"/opt/voxel/bin/fish2persp",
}

// Find will attempt to locate a fish2persp binary by looking directly on the path, looking in the
// standard voxel container installation location, and checking for a bazel artifact (in case of a test)
func Find() (string, error) {
	paths := fish2perspLocations
	runfile, err := bazel.Runfile("third_party/fish2persp/fish2persp")
	if err == nil {
		paths = append(paths, runfile)
	}

	for _, location := range paths {
		binpath, err := exec.LookPath(location)
		if err == nil {
			return binpath, nil
		}
	}

	return "", fmt.Errorf("failed to find fish2persp binary")
}

func writeBlankJPEG(ctx context.Context, filename string, width, height int) error {
	log.Ctx(ctx).Debug().Str("filename", filename).Int("w", width).Int("h", height).Msg("writeBlankJPEG")
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			img.Set(x, y, color.Black)
		}
	}

	outf, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to write blank image file: %w", err)
	}
	defer func() {
		if outf != nil {
			_ = outf.Close()
		}
	}()

	if err := jpeg.Encode(outf, img, nil); err != nil {
		return fmt.Errorf("failed to encode blank jpeg image: %w", err)
	}

	err = outf.Close()
	outf = nil

	if err != nil {
		return fmt.Errorf("error writing blank image file for: %w", err)
	}
	return nil
}

// RemapPGM contains the fish2persp_x.pgm and fish2persp_y.pgm file data
type RemapPGM struct {
	X []byte
	Y []byte
}

// GenerateRemapPGM generates remap pgm files for ffmpeg's remap filter
func GenerateRemapPGM(ctx context.Context, cfg *edgeconfigpb.Fish2PerspRemap) (*RemapPGM, error) {
	binpath, err := Find()
	if err != nil {
		return nil, err
	}

	tmpdir, err := os.MkdirTemp("", "fish2persp")
	if err != nil {
		return nil, fmt.Errorf("failed to create temporary directory: %w", err)
	}
	defer func() {
		_ = os.RemoveAll(tmpdir)
	}()

	sampleFilename := filepath.Join(tmpdir, "sample.jpg")
	sampleWidth := int(cfg.GetFish().WidthPixels)
	sampleHeight := int(cfg.GetFish().HeightPixels)
	if err := writeBlankJPEG(ctx, sampleFilename, sampleWidth, sampleHeight); err != nil {
		return nil, fmt.Errorf("failed to write blank sample image for fish2persp: %w", err)
	}

	args := []string{
		"-w", fmt.Sprintf("%d", cfg.Persp.WidthPixels),
		"-h", fmt.Sprintf("%d", cfg.Persp.HeightPixels),
		"-t", fmt.Sprintf("%d", cfg.Persp.FovDegrees),
		"-r", fmt.Sprintf("%d", cfg.Fish.RadiusXPixels),
		"-c", fmt.Sprintf("%d", cfg.Fish.CenterXPixels), fmt.Sprintf("%d", cfg.Fish.CenterYPixels),
		"-s", fmt.Sprintf("%d", cfg.Fish.FovDegrees),
		"-x", fmt.Sprintf("%d", cfg.Fish.TiltDegrees),
		"-y", fmt.Sprintf("%d", cfg.Fish.RollDegrees),
		"-z", fmt.Sprintf("%d", cfg.Fish.PanDegrees),
		"-f",
		sampleFilename,
	}

	// trunk-ignore(semgrep/go.lang.security.audit.dangerous-exec-command.dangerous-exec-command): all inputs are numerics
	cmd := exec.CommandContext(ctx, binpath, args...)
	cmd.Dir = tmpdir

	log.Ctx(ctx).Debug().Stringer("cmd", cmd).Msg("running fish2persp")

	// trunk-ignore(semgrep/trailofbits.go.invalid-usage-of-modified-variable.invalid-usage-of-modified-variable): false positive
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("fish2persp command failed: %w\n%s", err, string(out))
	}

	pgmx, err := os.ReadFile(filepath.Join(tmpdir, "fish2persp_x.pgm"))
	if err != nil {
		return nil, fmt.Errorf("failed to read fish2persp_x.pgm: %w", err)
	}

	pgmy, err := os.ReadFile(filepath.Join(tmpdir, "fish2persp_y.pgm"))
	if err != nil {
		return nil, fmt.Errorf("failed to read fish2persp_y.pgm: %w", err)
	}

	return &RemapPGM{X: pgmx, Y: pgmy}, nil
}
