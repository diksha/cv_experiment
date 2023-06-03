package transcoder

import (
	"strings"
)

// FFmpegFlags holds a set of ffmpeg flags that can be collected into a full set to run ffmpeg
type FFmpegFlags struct {
	Global Flags
	Input  Flags
	Filter Filter
	Encode Flags
	Output Flags
}

// Filter is a generic interface which allows for various kinds of filters to be used in ffmpeg
// Currently the only implementations are RawFilter and SimpleFilter, but we may add ComplexFilter
type Filter interface {
	Args() []string
	String() string
}

// Flags is a convenience type for a list of flags in a slice of strings
type Flags []string

// Append is a convenience function for appending to this flag list
func (f *Flags) Append(flags ...string) { *f = append(*f, flags...) }

// Prepend is a convenience function for prepending to this flag list
func (f *Flags) Prepend(flags ...string) { *f = append(flags, *f...) }

// Replace is a convenience function for replacing this flag list
func (f *Flags) Replace(flags ...string) { *f = flags }

// String constructs a string representation of this flag list
func (f *Flags) String() string { return strings.Join(*f, " ") }

// SimpleFilter is a simple ffmpeg filter graph
type SimpleFilter []string

// Append appends a set of filters to this filter
func (f *SimpleFilter) Append(filters ...string) {
	*f = append(*f, filters...)
}

// Prepend prepends a set of filters to this filter
func (f *SimpleFilter) Prepend(filters ...string) {
	*f = append(filters, *f...)
}

// Replace replaces this filter chain with the specified filters
func (f *SimpleFilter) Replace(filters ...string) {
	*f = filters
}

// String constructs a string representation of this simple filter set
func (f SimpleFilter) String() string { return strings.Join(f, ",") }

// Args produces the command line args for this simple filter
func (f SimpleFilter) Args() []string {
	return []string{"-vf", f.String()}
}

// NewSimpleFilter constructs a simple filter with the passed in filters
func NewSimpleFilter(filters ...string) SimpleFilter {
	return SimpleFilter(filters)
}

// CmdArgs converts this flag list to cmd args for execution
func (f *FFmpegFlags) CmdArgs() []string {
	args := []string{}
	if len(f.Global) > 0 {
		args = append(args, f.Global...)
	}

	if len(f.Input) > 0 {
		args = append(args, f.Input...)
	}

	args = append(args, f.Filter.Args()...)

	if len(f.Encode) > 0 {
		args = append(args, f.Encode...)
	}

	if len(f.Output) > 0 {
		args = append(args, f.Output...)
	}
	return args
}

// String constructs a string representation of this flag list
func (f *FFmpegFlags) String() string {
	return strings.Join(f.CmdArgs(), " ")
}

// RawFilter is a set of raw filter parameters that will be passed straight to ffmpeg
type RawFilter []string

// Args returns this set of flags unmodified
func (rf RawFilter) Args() []string { return []string(rf) }

// String returns a string representation of this raw filter
func (rf RawFilter) String() string { return strings.Join(rf, " ") }

// NewRawFilter constructs a new raw filter from the passed in flag and filter string
func NewRawFilter(flag, filter string) RawFilter { return RawFilter{flag, filter} }
