// Package iorate provides a library to enable rate limited reads/writes from a standard io.Reader/io.Writer
package iorate

import "fmt"

// Package iorate provides rate limited reader and writer implementations

// ByteRate is a data rate specified in Bytes per Second
type ByteRate int

var (
	// Unlimited is an unlimited byte rate
	Unlimited ByteRate
	// BPS is a byte rate of 1 Byte per second
	BPS ByteRate = 1
	// KBPS is a byte rate of 1 Kilobyte per second
	KBPS ByteRate = 1 << 10
	// MBPS is a byte rate of 1 Megabyte per second
	MBPS ByteRate = 1 << 20
	// GBPS is a byte rate of 1 Gigabyte per second
	GBPS ByteRate = 1 << 30
)

func (r ByteRate) String() string {
	switch {
	case r == Unlimited:
		return "unlimited"
	case r > GBPS:
		return fmt.Sprintf("%.3fgbps", float64(r)/float64(GBPS))
	case r > MBPS:
		return fmt.Sprintf("%.3fmbps", float64(r)/float64(MBPS))
	case r > KBPS:
		return fmt.Sprintf("%.3fkbps", float64(r)/float64(KBPS))
	default:
		return fmt.Sprintf("%dbps", r)
	}
}
