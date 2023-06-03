package ffmpeg

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// Cmd holds a handle to an ffmpeg process with some
// configuration aids. This is must be constructed with New()
type Cmd struct {
	// Args is the set of flags which will be passed to ffmpeg
	Args []string
	// LogLevel will be passed to the `-loglevel` flag
	LogLevel string

	progress chan Progress
	output   chan string
	done     chan struct{}
	cmderr   error

	// Cmd holds the handle to ffmpeg's os process, will only be populated
	// after Start() is called
	*exec.Cmd
}

// Progress holds values returned by ffmpeg's progress flag, which is unfortunately
// not very well documented.
type Progress struct {
	Frame         int64
	Fps           float64
	BitrateString string
	TotalSize     int64

	// Output timestamp of the last frame
	OutTimestamp time.Time

	DupFrames  int64
	DropFrames int64
	Speed      float64
	ExtraData  []string
}

// New constructs a Cmd with the specified context and args
func New(ctx context.Context, args ...string) (*Cmd, error) {
	cmd := &Cmd{
		Args: args,
	}
	ffmpegBin, err := FindFFmpeg()
	if err != nil {
		return nil, fmt.Errorf("failed to find ffmpeg: %w", err)
	}
	cmd.Cmd = exec.CommandContext(ctx, ffmpegBin)
	return cmd, nil
}

// Start constructs and starts a Cmd
func Start(ctx context.Context, args ...string) (*Cmd, error) {
	cmd, err := New(ctx, args...)
	if err != nil {
		return nil, err
	}

	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return cmd, nil
}

// Progress can be read to receive progress data from the `-progress flag`
func (cmd *Cmd) Progress() <-chan Progress {
	return cmd.progress
}

// Output can be read to retrieve output log data from ffmpeg, stores up to
// 1000 lines before it starts dropping data. Will be nil if Cmd.Stderr is set
func (cmd *Cmd) Output() <-chan string {
	return cmd.output
}

// Done returns a channel which will be closed once the program exits
func (cmd *Cmd) Done() <-chan struct{} {
	return cmd.done
}

// Wait will block until the command has exited and then return the exit error
func (cmd *Cmd) Wait() error {
	<-cmd.Done()
	return cmd.Err()
}

// Err returns the error value if any returned by Wait()
func (cmd *Cmd) Err() error {
	return cmd.cmderr
}

// Run starts this command and then waits for it to exit, returning the exit value
func (cmd *Cmd) Run() error {
	if err := cmd.Start(); err != nil {
		return err
	}
	return cmd.Wait()
}

func (cmd *Cmd) args(progressFd int) []string {
	logLevel := cmd.LogLevel
	if logLevel == "" {
		logLevel = "error"
	}

	args := append(cmd.Cmd.Args, []string{
		"-loglevel", logLevel,
		"-nostdin",
		"-hide_banner",
		"-progress", fmt.Sprintf("pipe:%d", progressFd),
	}...)

	args = append(args, cmd.Args...)
	return args
}

// returns a string representation of this command for debugging purposes
func (cmd *Cmd) String() string {
	// we pick a progressfd value of 3 as it is the most likely value, and this is just for debugging
	return strings.Join(cmd.args(3), " ")
}

// Start attempts to start ffmpeg with the specified configs
func (cmd *Cmd) Start() (err error) {
	progressReader, progressWriter, err := os.Pipe()
	if err != nil {
		return fmt.Errorf("failed to create ffmpeg progress pipe: %w", err)
	}
	defer func() {
		if err != nil {
			_ = progressReader.Close()
			_ = progressWriter.Close()
		}
	}()

	cmd.Cmd.ExtraFiles = append(cmd.Cmd.ExtraFiles, progressWriter)
	progressFd := len(cmd.Cmd.ExtraFiles) + 2

	cmd.Cmd.Args = cmd.args(progressFd)

	progressCh := make(chan Progress, 1)
	go ReadProgress(progressCh, progressReader)
	cmd.progress = progressCh

	if cmd.Cmd.Stderr == nil {
		stderr, err := cmd.Cmd.StderrPipe()
		if err != nil {
			return fmt.Errorf("failed to get ffmpeg stderr pipe: %w", err)
		}

		// read output from stderr and push it into outputCh, default to a 1000 line buffer
		outputCh := make(chan string, 1000)
		go func() {
			defer close(outputCh)
			scanner := bufio.NewScanner(stderr)
			for scanner.Scan() {
				select {
				case outputCh <- scanner.Text():
				default:
				}
			}
		}()
		cmd.output = outputCh
	}

	if err = cmd.Cmd.Start(); err != nil {
		return fmt.Errorf("failed to start ffmpeg: %w", err)
	}

	cmd.done = make(chan struct{})
	go func() {
		defer close(cmd.done)
		cmd.cmderr = cmd.Cmd.Wait()
	}()

	// we can close our handle on the write end of the progress writer pipe
	// which will ensure that the read end closes when ffmpeg exits
	_ = progressWriter.Close()

	return nil
}

// ReadProgress can read key=value formatted outputs from
// ffmpeg's `-progress` flag into a Progress struct.
func ReadProgress(outCh chan<- Progress, r io.Reader) {
	defer close(outCh)
	scanner := bufio.NewScanner(r)

	var progress *Progress
	for scanner.Scan() {
		if len(scanner.Text()) == 0 {
			// skip empty lines
			continue
		}

		if progress == nil {
			//make sure we have somewhere to store the value we read
			progress = &Progress{}
		}

		spl := strings.Split(scanner.Text(), "=")
		if len(spl) != 2 {
			// unrecognizable data just gets stored
			progress.ExtraData = append(progress.ExtraData, scanner.Text())
			continue
		}

		consumed := false
		switch spl[0] {
		case "frame":
			// values like "frame=134"
			if v, err := strconv.ParseInt(spl[1], 10, 64); err == nil {
				progress.Frame = v
				consumed = true
			}
		case "fps":
			// values like "fps=5.31"
			if v, err := strconv.ParseFloat(spl[1], 64); err == nil {
				progress.Fps = v
				consumed = true
			}
		case "bitrate":
			// values like "bitrate= 448.1kbits/s"
			progress.BitrateString = strings.TrimSpace(spl[1])
			consumed = true
		case "total_size":
			// values like "total_size=1310720"
			if v, err := strconv.ParseInt(spl[1], 10, 64); err == nil {
				progress.TotalSize = v
				consumed = true
			}
		case "out_time_us":
			// values like "out_time_us=23401000"
			if outTimeMicros, err := strconv.ParseInt(spl[1], 10, 64); err == nil {
				progress.OutTimestamp = time.UnixMicro(outTimeMicros)
				consumed = true
			}
		case "dup_frames":
			// values like "dup_frames=0"
			if v, err := strconv.ParseInt(spl[1], 10, 64); err == nil {
				progress.DupFrames = v
				consumed = true
			}
		case "drop_frames":
			// values like "drop_frames=0"
			if v, err := strconv.ParseInt(spl[1], 10, 64); err == nil {
				progress.DropFrames = v
				consumed = true
			}
		case "speed":
			// values like "speed=0.928x"
			speedStr := strings.TrimSuffix(strings.TrimSpace(spl[1]), "x")
			if v, err := strconv.ParseFloat(speedStr, 64); err == nil {
				progress.Speed = v
				consumed = true
			}
		case "progress":
			// we have hit the next progress entry, emit this value
			if progress != nil {
				select {
				case outCh <- *progress:
				default:
				}
				progress = nil
			}
			consumed = true
		}

		if !consumed {
			progress.ExtraData = append(progress.ExtraData, scanner.Text())
		}
	}

	// we have exited the scan loop, check if there was a
	// progress update left in the buffer and emit it if so
	if progress != nil {
		select {
		case outCh <- *progress:
		default:
		}
	}
}
