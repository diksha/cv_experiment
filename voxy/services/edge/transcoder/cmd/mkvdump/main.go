package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"time"

	"github.com/at-wat/ebml-go"
	"github.com/remko/go-mkvparse"
)

func init() {
	flag.Usage = func() {
		w := flag.CommandLine.Output()
		fmt.Fprintf(w, "Usage of %s <input>\n", os.Args[0])
		fmt.Fprintf(w, "\n")
		fmt.Fprintf(w, "\tThis program reads a stream of matroska tags and displays their content. Input must be a file (or - for stdin).\n")
		flag.PrintDefaults()
	}
}

type handler struct {
	indent           int
	timebase         int64
	clusterTimestamp int64
}

func (h *handler) Logf(format string, args ...interface{}) {
	log.Printf(strings.Repeat("\t", h.indent)+format, args...)
}

// Returns `true` (recurses into the master element)
func (h *handler) HandleMasterBegin(id mkvparse.ElementID, info mkvparse.ElementInfo) (bool, error) {
	h.Logf("HandleMasterBegin(%s, %+v)", mkvparse.NameForElementID(id), info)
	h.indent++

	switch id {
	case mkvparse.InfoElement:
		fallthrough
	case mkvparse.ClusterElement:
		fallthrough
	case mkvparse.SegmentElement:
		return true, nil
	}

	return true, nil
}

func (h *handler) HandleMasterEnd(id mkvparse.ElementID, info mkvparse.ElementInfo) error {
	h.indent--
	h.Logf("HandleMasterEnd(%s, %+v)", mkvparse.NameForElementID(id), info)
	return nil
}

func (h *handler) HandleString(id mkvparse.ElementID, value string, info mkvparse.ElementInfo) error {
	h.Logf("HandleString(%s, %q, %+v)", mkvparse.NameForElementID(id), value, info)
	return nil
}

func (h *handler) HandleInteger(id mkvparse.ElementID, value int64, info mkvparse.ElementInfo) error {
	h.Logf("HandleInteger(%s, %v, %+v)", mkvparse.NameForElementID(id), value, info)
	if id == mkvparse.TimecodeScaleElement {
		h.timebase = value
	}
	if id == mkvparse.TimecodeElement {
		h.clusterTimestamp = value
	}
	return nil
}

func (h *handler) HandleFloat(id mkvparse.ElementID, value float64, info mkvparse.ElementInfo) error {
	h.Logf("HandleFloat(%s, %v, %+v)", mkvparse.NameForElementID(id), value, info)
	return nil
}

func (h *handler) HandleDate(id mkvparse.ElementID, value time.Time, info mkvparse.ElementInfo) error {
	h.Logf("HandleDate(%s, %v, %+v)", mkvparse.NameForElementID(id), value, info)
	return nil
}

func (h *handler) HandleBinary(id mkvparse.ElementID, value []byte, info mkvparse.ElementInfo) error {
	h.Logf("HandleBinary(%s, bytes len(%d), %+v)", mkvparse.NameForElementID(id), len(value), info)
	if id == mkvparse.SimpleBlockElement {
		block, err := ebml.UnmarshalBlock(bytes.NewReader(value), int64(len(value)))
		if err != nil {
			return fmt.Errorf("error parsing simple block: %w", err)
		}
		h.Logf("TrackNumber=%d Timecode=%d", block.TrackNumber, block.Timecode)
	}
	return nil
}

type readerOnly struct {
	io.Reader
}

func main() {
	log.SetFlags(0)
	flag.Parse()
	if flag.NArg() == 0 {
		flag.Usage()
		os.Exit(1)
	}

	var mkvR io.Reader
	inputFilename := flag.Arg(0)
	if inputFilename == "-" {
		log.Printf("Reading from stdin")
		mkvR = readerOnly{os.Stdin}
	} else {

		log.Printf("Opening file %q", inputFilename)
		f, err := os.Open(inputFilename)
		if err != nil {
			log.Fatalf("Failed to open input file %q: %v", inputFilename, err)
		}
		defer func() { _ = f.Close() }()
		mkvR = f
	}

	log.Fatal(mkvparse.Parse(mkvR, &handler{}))
}
