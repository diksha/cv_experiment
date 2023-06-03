package logtailer

import (
	"fmt"
	"io"
	"log"
	"path/filepath"
	"sync"
	"time"

	"github.com/nxadm/tail"
)

// Line is a log line from one of the files being watched by this package
type Line struct {
	Filename string
	Text     string
}

// Watcher is a handle returned by Watch which which allows it to be stopped
type Watcher struct {
	wg       sync.WaitGroup
	stopOnce sync.Once
	stop     chan struct{}
	ch       chan *Line
	refresh  chan struct{}
}

// Watch reads all lines from the files matching the passed in globs, reopening
// as necessary while also watching for new files as they are created
func Watch(globs ...string) (watcher *Watcher, err error) {
	var allfiles []string
	for _, glob := range globs {
		filenames, err := filepath.Glob(glob)
		if err != nil {
			return nil, fmt.Errorf("failed to search glob %q: %w", glob, err)
		}
		allfiles = append(allfiles, filenames...)
	}

	tails := make(map[string]*tail.Tail)
	watcher = &Watcher{
		ch:      make(chan *Line),
		stop:    make(chan struct{}),
		refresh: make(chan struct{}, 1),
	}

	defer func() {
		if err != nil {
			// release all resources on startup failure
			for _, tail := range tails {
				_ = tail.Stop()
				tail.Cleanup()
			}
		}
	}()

	// the first time we open any files we will be opening them at the end
	// this is to avoid re-reading data during a restart. in practice this
	// means we will lose a small amount of data when this component restarts
	// but that should be infrequent enough that it really should not matter
	for _, filename := range allfiles {
		if err := watcher.startTail(tails, filename, &tail.SeekInfo{Whence: io.SeekEnd}); err != nil {
			return nil, err
		}
	}

	go watcher.pollGlobs(tails, globs)

	return watcher, nil
}

// Refresh triggers an immediate refresh of the files this watcher is watching
// This is mostly for here testing/debug purposes
func (w *Watcher) Refresh() {
	select {
	case w.refresh <- struct{}{}:
	default:
	}
}

// Stop stops this watcher and releases its resources
func (w *Watcher) Stop() {
	w.stopOnce.Do(func() {
		close(w.stop)
	})
}

func (w *Watcher) pollGlobs(tails map[string]*tail.Tail, globs []string) {
	// poll for new files once every 5 seconds
	// trunk-ignore(semgrep/trailofbits.go.nondeterministic-select.nondeterministic-select): doesn't matter in this case
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	// we use multiple defers for this exit logic to try and guarantee that it all runs
	// even if there are errors. defers run in LIFO order so they are reversed here.
	defer close(w.ch)
	defer w.wg.Wait()
	defer func() {
		for _, tail := range tails {
			_ = tail.Stop()
			tail.Cleanup()
		}
	}()

	for {
		var allfiles []string
		for _, glob := range globs {
			filenames, err := filepath.Glob(glob)
			if err != nil {
				// not sure what the best course of action here is so
				// we will just log the error and continue for now
				log.Printf("failed to find files for glob %q: %v", glob, err)
				continue
			}

			allfiles = append(allfiles, filenames...)
		}

		w.checkTails(tails, allfiles)

		select {
		case <-w.stop:
			return
		case <-ticker.C:
		case <-w.refresh:
		}
	}
}

func (w *Watcher) checkTails(tails map[string]*tail.Tail, allfiles []string) {
	for filename, t := range tails {
		if !StringSliceContains(allfiles, filename) {
			// one of our tails no longer exists, remove it
			delete(tails, filename)
			go func(tailToClose *tail.Tail) {
				// we do this in a goroutine because stop can block and we
				// don't need to guarantee that the tail stops within
				// a certain amount of time
				_ = tailToClose.Stop()
				tailToClose.Cleanup()
			}(t)
		}
	}

	for _, filename := range allfiles {
		if _, ok := tails[filename]; !ok {
			// this is a new file, we start tailing it from the beginning
			err := w.startTail(tails, filename, nil)
			if err != nil {
				log.Printf("failed to start tailing file %q: %v", filename, err)
			}
		}
	}
}

func (w *Watcher) startTail(tails map[string]*tail.Tail, filename string, location *tail.SeekInfo) error {
	tailer, err := tail.TailFile(filename, tail.Config{
		MustExist: true,
		Follow:    true,
		Location:  location,
		ReOpen:    true,
	})
	if err != nil {
		return fmt.Errorf("failed to tail file %q: %w", filename, err)
	}

	// we waited to lock this mutex until right before we use it
	tails[filename] = tailer

	w.wg.Add(1)
	go func() {
		defer w.wg.Done()
		w.tail(filename, tailer)
	}()

	return nil
}

func (w *Watcher) tail(filename string, t *tail.Tail) {
	for line := range t.Lines {
		if line.Err != nil {
			return
		}
		w.ch <- &Line{
			Filename: filename,
			Text:     line.Text,
		}
	}
}

// Lines is a channel of *Line elements from the files this watcher is watching
func (w *Watcher) Lines() <-chan *Line {
	return w.ch
}

// StringSliceContains returns true if the string s is in the slice
func StringSliceContains(slice []string, s string) bool {
	m := make(map[string]struct{})
	for _, val := range slice {
		m[val] = struct{}{}
	}

	_, ok := m[s]
	return ok
}
