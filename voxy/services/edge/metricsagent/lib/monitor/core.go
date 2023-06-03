package monitor

import (
	"context"
	"fmt"
	"time"

	"github.com/shirou/gopsutil/cpu"
	"github.com/shirou/gopsutil/disk"
	"github.com/shirou/gopsutil/load"
	"github.com/shirou/gopsutil/mem"

	"github.com/voxel-ai/voxel/go/edge/metricsctx"
)

const (
	metricKeyCPUUsage          = "CPUUsage"
	metricKeyMemoryUsage       = "MemoryUsage"
	metricKeyLoad1             = "Load1"
	metricKeyMaxPartitionUsage = "MaxPartitionUsage"
)

// AverageLoad monitors the average load on a device
func AverageLoad(ctx context.Context, interval time.Duration) error {
	for {
		loadstat, err := load.Avg()
		if err != nil {
			return fmt.Errorf("failed to measure load: %w", err)
		}

		metricsctx.Publish(ctx, metricKeyLoad1, time.Now(), metricsctx.UnitNone, loadstat.Load1, nil)

		time.Sleep(interval)
	}
}

// MemoryUsage monitors the percentage of RAM which a device is utilizing
func MemoryUsage(ctx context.Context, interval time.Duration) error {
	for {
		vmstat, err := mem.VirtualMemory()
		if err != nil {
			return fmt.Errorf("failed to measure memory: %w", err)
		}

		metricsctx.Publish(ctx, metricKeyMemoryUsage, time.Now(), metricsctx.UnitPercent, vmstat.UsedPercent, nil)

		time.Sleep(interval)
	}
}

// CPUUsage monitors the percentage a device's CPU that is in use
func CPUUsage(ctx context.Context, interval time.Duration) error {
	for {
		values, err := cpu.Percent(interval, false)
		if err != nil {
			return fmt.Errorf("failed to measure cpu usage: %w", err)
		}

		metricsctx.Publish(ctx, metricKeyCPUUsage, time.Now(), metricsctx.UnitPercent, values[0], nil)
	}
}

// DiskUsage monitors the percentage of disk space a device is using
func DiskUsage(ctx context.Context, interval time.Duration) error {
	paths := []string{
		"/",
		"/var",
		"/var/log",
	}

	for {
		var maxUsed float64
		for _, path := range paths {
			used, err := diskUsedPercent(path)
			if err != nil {
				return fmt.Errorf("error monitoring disk usage: %w", err)
			}
			if used > maxUsed {
				maxUsed = used
			}
		}

		metricsctx.Publish(ctx, metricKeyMaxPartitionUsage, time.Now(), metricsctx.UnitPercent, maxUsed, nil)

		time.Sleep(interval)
	}
}

func diskUsedPercent(path string) (float64, error) {
	ustat, err := disk.Usage(path)
	if err != nil {
		return 0, fmt.Errorf("failed to get disk usage for %q: %w", path, err)
	}

	return ustat.UsedPercent, nil
}
