package gstreamer

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// regex matchers used by parseGSTStatsTracer
var gstTraceRegex = regexp.MustCompile(`^.*GST_TRACER\s+.*:.*:.*:\S*\s+(.*);`)
var gstTraceMessageRegex = regexp.MustCompile(`(.+?)(?:, |$)`)

// Cmd holds a gstreamer process configuration and handles for controlling the process lifecycle.
// New Cmd values should be constructed with New
type Cmd struct {
	// Cmd will be populated by
	*exec.Cmd

	stats  chan Stats
	output chan string

	done   chan struct{}
	cmderr error
}

// New sets up a new Cmd value with the passed in context and
// uses the file handle as the stdin for the process
func New(ctx context.Context, args ...string) *Cmd {
	cmd := exec.CommandContext(ctx, "gst-launch-1.0", args...)
	cmd.Env = os.Environ()
	cmd.Env = append(cmd.Env, []string{
		"GST_DEBUG=2,GST_TRACER:7",
		"GST_TRACERS=stats",
		"GST_DEBUG_NO_COLOR=1",
	}...)
	return &Cmd{
		Cmd: cmd,
	}
}

// Start appends args to the exec.Cmd and starts it, returning errors if the arguments are invalid or start fails
func (cmd *Cmd) Start() error {
	// this will be used as an output channel for stderr/stdout
	outCh := make(chan string, 1000)
	if cmd.Cmd.Stderr == nil {
		// set up a pipe to read stats and open it
		stderr, err := cmd.Cmd.StderrPipe()
		if err != nil {
			return fmt.Errorf("failed to open gstreamer stderr pipe: %w", err)
		}

		// we make an io pipe (which is not an os pipe) so that we can use a tee reader
		// to allow the stats reader and the logs reader to both read stderr from gstreamer
		statsReader, statsWriter := io.Pipe()

		statsCh := make(chan Stats, 1)
		go ReadStats(statsCh, io.TeeReader(stderr, statsWriter))
		cmd.stats = statsCh

		// store up to 1000 log lines
		go ReadLogs(outCh, statsReader)
		cmd.output = outCh

	}

	if cmd.Cmd.Stdout == nil {
		stdout, err := cmd.StdoutPipe()
		if err != nil {
			return fmt.Errorf("failed to open gstreamer stdout pipe: %w", err)
		}

		go ReadLogs(outCh, stdout)
		cmd.output = outCh
	}

	if err := cmd.Cmd.Start(); err != nil {
		return fmt.Errorf("failed to start gstreamer process: %w", err)
	}

	cmd.done = make(chan struct{})
	go func() {
		defer close(cmd.done)
		cmd.cmderr = cmd.Cmd.Wait()
	}()

	return nil
}

// Stats will receive stats updates from gstreamer if Stderr is not overridden
func (cmd *Cmd) Stats() <-chan Stats {
	return cmd.stats
}

// Output will receive log lines from stderr if it has not been overridden
func (cmd *Cmd) Output() <-chan string {
	return cmd.output
}

// Wait blocks until the underlying process exits and returns any error occurred
func (cmd *Cmd) Wait() error {
	<-cmd.Done()
	return cmd.Err()
}

// Run is a convenience funciton that calls start and wait
func (cmd *Cmd) Run() error {
	if err := cmd.Start(); err != nil {
		return err
	}

	return cmd.Wait()
}

// Done returns a channel that will be closed once the process exits
func (cmd *Cmd) Done() <-chan struct{} {
	return cmd.done
}

// Err returns the error returned by wait, if any
func (cmd *Cmd) Err() error {
	return cmd.cmderr
}

// Stats holds stats information parsed from the stats tracer in gstreamer
type Stats struct {
	// This is the time of the last observed buffer
	LastBufferTimestamp time.Duration
}

// ReadLogs reads lines from the passed in reader and drops them into the logging channel
func ReadLogs(outCh chan string, r io.Reader) {
	defer close(outCh)
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		if strings.Contains(scanner.Text(), "GST_TRACER") {
			// ignore GST_TRACER lines as they aren't very helpful in logs
			continue
		}
		select {
		case outCh <- scanner.Text():
		default:
			// write would block, perform a non-blocking read to pop the oldest log
			// off the queue and then try again
			select {
			case <-outCh:
			default:
			}
			// retrying the non-blocking write
			select {
			case outCh <- scanner.Text():
			default:
			}
		}
	}
	if scanner.Text() != "" {
		select {
		case outCh <- scanner.Text():
		default:
		}
	}
}

// ReadStats attempts to parse buffer timestamps from gstreamer GST_TRACE lines.
// This function is used internally by Cmd and should not normally need to be called.
func ReadStats(outCh chan<- Stats, r io.Reader) {
	defer close(outCh)
	var stats Stats
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		messageMatch := gstTraceRegex.FindStringSubmatch(scanner.Text())
		if len(messageMatch) < 2 {
			continue
		}
		kvpairsMatch := gstTraceMessageRegex.FindAllStringSubmatch(messageMatch[1], -1)
		if len(kvpairsMatch) == 0 || len(kvpairsMatch[0]) < 2 || kvpairsMatch[0][1] != "buffer" {
			// we only care about buffer messages
			continue
		}
		kvpairs := make(map[string]string)
		for _, m := range kvpairsMatch {
			if len(m) < 2 {
				continue
			}
			spl := strings.SplitN(m[1], "=", 2)
			if len(spl) < 2 {
				continue
			}
			kvpairs[spl[0]] = spl[1]
		}

		// the original value is a uint64 but we are parsing into an int64 because
		// golang time.Duration values are int64 and not uint64
		tsInt, err := strconv.ParseInt(strings.TrimPrefix(kvpairs["ts"], "(guint64)"), 10, 64)
		// conversion from a raw timestamp to a Go time.Duration is 1:1 as time.Duration is in nanos
		if err == nil {
			ts := time.Duration(tsInt)
			// looks like we actually got a new timestamp value
			if ts > stats.LastBufferTimestamp {
				stats.LastBufferTimestamp = ts
				// do a non-blocking write to the stats channel
				select {
				case outCh <- stats:
				default:
				}
			}
		}
	}
}
