package kvspusher

import (
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/at-wat/ebml-go"
	"github.com/at-wat/ebml-go/mkv"
)

// DefaultMKVHeader holds default values for the header fields of an MKV file
var DefaultMKVHeader = mkv.EBMLHeader{
	EBMLVersion:        1,
	EBMLReadVersion:    1,
	EBMLMaxIDLength:    4,
	EBMLMaxSizeLength:  8,
	DocType:            "matroska",
	DocTypeVersion:     2,
	DocTypeReadVersion: 2,
}

// TrackEntry represents the MKV TrackEntry type but only includes the components necessary to work with Kinesis Video
type TrackEntry struct {
	TrackNumber     uint64
	TrackUID        uint64
	FlagLacing      uint64
	CodecID         string
	TrackType       uint64
	DefaultDuration uint64
	Video           struct {
		PixelWidth  uint64
		PixelHeight uint64
	}
	CodecPrivate []byte
}

// Info represents the MKV Info type but only includes the components necessary to work with Kinesis Video
type Info struct {
	TimecodeScale   uint64
	SegmentUID      []byte
	SegmentFilename string
	Title           string
	MuxingApp       string
	WritingApp      string
}

// Cluster represents the MKV Cluster type but only includes the components necessary to work with Kinesis Video
type Cluster struct {
	Timecode    uint64
	SimpleBlock []ebml.Block
}

// Segment represents the MKV Segment type but only includes the components necessary to work with Kinesis Video
type Segment struct {
	Info   Info
	Tracks struct {
		TrackEntry []TrackEntry
	}
	Cluster []Cluster
}

// Fragment represents a valid MKV fragment
type Fragment struct {
	EBML    mkv.EBMLHeader
	Segment Segment
}

// BlockList is a utility type to enable `sort.Sort`
type BlockList []ebml.Block

func (bl BlockList) Len() int           { return len(bl) }
func (bl BlockList) Less(i, j int) bool { return bl[i].Timecode < bl[j].Timecode }
func (bl BlockList) Swap(i, j int)      { bl[i], bl[j] = bl[j], bl[i] }

// ReadFragment loads an MKV fragment from the passed in reader
func ReadFragment(r io.Reader) (*Fragment, error) {
	var f Fragment
	if err := ebml.Unmarshal(r, &f); err != nil {
		return nil, fmt.Errorf("failed to unmarshal mkv fragment: %w", err)
	}
	return &f, nil
}

// WriteFragment writes a marshaled MKV fragment to the passed in writer
func WriteFragment(w io.Writer, f *Fragment) error {
	if err := ebml.Marshal(f, w); err != nil {
		return fmt.Errorf("failed to marshal mkv fragment: %w", err)
	}
	return nil
}

// MergeClusters combines all passed in clusters into a single cluster
func MergeClusters(clusters []Cluster) Cluster {
	var out Cluster

	if len(clusters) == 1 {
		return clusters[0]
	}

	// find input cluster with the lowest timestamp
	for _, c := range clusters {
		if out.Timecode == 0 || c.Timecode < out.Timecode {
			out.Timecode = c.Timecode
		}
	}

	// insert all blocks into the single cluster we are building
	for _, c := range clusters {
		delta := int16(c.Timecode - out.Timecode)
		for _, b := range c.SimpleBlock {
			b.Timecode += delta
			out.SimpleBlock = append(out.SimpleBlock, b)
		}
	}

	return out
}

// SimplifyFragment removes fragment components which are not needed by kinesis and merges all clusters to a single cluster
func SimplifyFragment(inf *Fragment) *Fragment {
	outf := &Fragment{
		EBML: DefaultMKVHeader,
		Segment: Segment{
			Info:    inf.Segment.Info,
			Tracks:  inf.Segment.Tracks,
			Cluster: []Cluster{MergeClusters(inf.Segment.Cluster)},
		},
	}
	return outf
}

// GetAllTimestamps returns a list of all timestamps in the fragment
func (frag *Fragment) GetAllTimestamps() []time.Duration {
	var timestamps []time.Duration

	for _, cluster := range frag.Segment.Cluster {
		for _, block := range cluster.SimpleBlock {
			timestampAsDuration := time.Duration((uint64(block.Timecode)+cluster.Timecode)*frag.Segment.Info.TimecodeScale) * time.Nanosecond
			timestamps = append(timestamps, timestampAsDuration)
		}
	}
	return timestamps
}

// MinTimestamp returns the earliest timestamp in the fragment
func (frag *Fragment) MinTimestamp() (time.Duration, error) {
	allTimestamps := frag.GetAllTimestamps()
	if len(allTimestamps) == 0 {
		return 0, errors.New("no timestamps found in fragment")
	}

	minTimestamp := allTimestamps[0]
	for _, timestamp := range frag.GetAllTimestamps() {
		if timestamp < minTimestamp {
			minTimestamp = timestamp
		}
	}
	return minTimestamp, nil
}

// MaxTimestamp returns the latest timestamp in the fragment
func (frag *Fragment) MaxTimestamp() (time.Duration, error) {
	allTimestamps := frag.GetAllTimestamps()
	if len(allTimestamps) == 0 {
		return 0, errors.New("no timestamps found in fragment")
	}

	maxTimestamp := allTimestamps[0]
	for _, timestamp := range frag.GetAllTimestamps() {
		if timestamp > maxTimestamp {
			maxTimestamp = timestamp
		}
	}
	return maxTimestamp, nil
}

// Duration returns the duration of the fragment
func (frag *Fragment) Duration() (time.Duration, error) {
	minTimestamp, err := frag.MinTimestamp()
	if err != nil {
		return 0, err
	}
	maxTimestamp, err := frag.MaxTimestamp()
	if err != nil {
		return 0, err
	}
	return maxTimestamp - minTimestamp, nil
}
