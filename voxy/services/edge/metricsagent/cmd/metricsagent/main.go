// Executable for collecting metrics and logs from the edge and publishing them to cloudwatch.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/cloudwatch"

	"github.com/voxel-ai/voxel/go/edge/metricsctx"
	"github.com/voxel-ai/voxel/services/edge/metricsagent/lib/monitor"
)

// EdgeType is the edge hardware type, like quicksync or cuda
type EdgeType string

var (
	// EdgeTypeQuicksync is any generation of quicksync hardware that responds to intel_gpu_top
	EdgeTypeQuicksync EdgeType = "quicksync"
	// EdgeTypeCUDA is any nvidia cuda hardware that supports nvenc
	EdgeTypeCUDA EdgeType = "cuda"
	// EdgeTypeTest is used for testing purposes
	EdgeTypeTest EdgeType = "test"
)

var edgeUUID = flag.String("edge-uuid", "", "edge uuid for metrics tagging")
var edgeTypeString = flag.String("edge-type", "", "edge type (quicksync, cuda)")
var dryRun = flag.Bool("dry-run", false, "dry run skips publishing metrics to aws and logs instead")
var edgeType EdgeType

const (
	dimensionEdgeUUID = "EdgeUUID"

	monitorInterval = 10 * time.Second
)

func getCloudwatchClient(ctx context.Context) (*cloudwatch.Client, error) {
	loadCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	config, err := config.LoadDefaultConfig(loadCtx)
	if err != nil {
		return nil, fmt.Errorf("failed to load default aws config: %w", err)
	}
	return cloudwatch.NewFromConfig(config), nil
}

func parseArguments() {
	flag.Parse()

	edgeType = EdgeType(*edgeTypeString)

	switch edgeType {
	case EdgeTypeQuicksync:
	case EdgeTypeCUDA:
	case EdgeTypeTest:
	default:
		log.Fatal("-edge-type must be quicksync, cuda or test")
	}
}

func main() {
	ctx := context.Background()

	parseArguments()

	cloudwatchClient, err := getCloudwatchClient(ctx)
	if err != nil {
		log.Fatal(err)
	}

	metricsConfig := metricsctx.Config{
		Client:    cloudwatchClient,
		Interval:  1 * time.Minute,
		Namespace: fmt.Sprintf("voxel/edge/%v", edgeType),
		DryRun:    *dryRun,
		Dimensions: metricsctx.Dimensions{
			dimensionEdgeUUID: *edgeUUID,
		},
	}

	ctx = metricsctx.WithPublisher(ctx, metricsConfig)

	go publishLogFiles(ctx)

	go func() {
		log.Fatalf("CPU usage monitor exited with: %v", monitor.CPUUsage(ctx, monitorInterval))
	}()

	go func() {
		log.Fatalf("Memory usage monitor exited with: %v", monitor.MemoryUsage(ctx, monitorInterval))
	}()

	go func() {
		log.Fatalf("Load monitor exited with: %v", monitor.AverageLoad(ctx, monitorInterval))
	}()

	go func() {
		log.Fatalf("Disk usage monitor exited with: %v", monitor.DiskUsage(ctx, monitorInterval))
	}()

	go func() {
		log.Fatalf("Network stats monitor exited with: %v", monitor.NetworkStats(ctx, monitorInterval))
	}()

	switch edgeType {
	case EdgeTypeQuicksync:
		go func() {
			log.Fatalf("GPU usage monitor exited with: %v", monitor.GPUUsageQuicksync(ctx, monitorInterval))
		}()
	case EdgeTypeCUDA:
		go func() {
			log.Fatalf("GPU usage monitor exited with: %v", monitor.GPUUsageCUDA(ctx, monitorInterval))
		}()
	}

	<-ctx.Done()
}
