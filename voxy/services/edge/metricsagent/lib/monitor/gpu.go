package monitor

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/voxel-ai/voxel/go/edge/metricsctx"
)

const (
	metricKeyQuicksyncGPUVideoUsage  = "GPUVideoUsage"
	metricKeyQuicksyncGPURenderUsage = "GPURenderUsage"

	metricKeyCUDAGPUUsage        = "GPUUsage"
	metricKeyCUDAGPUEncoderUsage = "GPUEncoderUsage"
	metricKeyCUDAGPUDecoderUsage = "GPUDecoderUsage"
)

type intelGPUTopStats struct {
	Engines map[string]struct {
		Busy float64 `json:"busy"`
		Sema float64 `json:"sema"`
		Wait float64 `json:"wait"`
		Unit string  `json:"unit"`
	}
}

func parseGPUQuicksyncMetrics(gpuStatReader io.Reader, statsChannel chan<- intelGPUTopStats) error {
	defer close(statsChannel)

	// the output from intel_gpu_top is a comma delimited series of json objects
	// so we are going to pre-pend an open bracket, read it with Token(), and then
	// read the stream of objects from the decoder
	decoder := json.NewDecoder(io.MultiReader(bytes.NewReader([]byte("[")), gpuStatReader))

	// parse the open bracket
	if _, err := decoder.Token(); err != nil {
		return fmt.Errorf("failed to parse array delimiter from intel_gpu_top: %w", err)
	}

	for decoder.More() {
		var stats intelGPUTopStats
		if err := decoder.Decode(&stats); err != nil {
			return fmt.Errorf("failed to unmarshal intel_gpu_top output: %w", err)
		}

		statsChannel <- stats
	}

	return nil
}

// GPUUsageQuicksync monitors the GPU usage for video and rendering on a quicksync device
func GPUUsageQuicksync(ctx context.Context, interval time.Duration) error {
	cmdCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	durationArg := fmt.Sprintf("%v", interval.Milliseconds())

	cmd := exec.CommandContext(cmdCtx, "intel_gpu_top", "-J", "-s", durationArg)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to set up stdout pipe for intel_gpu_top: %w", err)
	}

	if err = cmd.Start(); err != nil {
		return fmt.Errorf("failed to start intel_gpu_top: %w", err)
	}

	errChannel := make(chan error, 2)
	statsChannel := make(chan intelGPUTopStats)

	go func(errCh chan error) {
		errCh <- fmt.Errorf("intel_gpu_top exited with: %w", cmd.Wait())
	}(errChannel)

	go func(errCh chan error) {
		errChannel <- parseGPUQuicksyncMetrics(stdout, statsChannel)
	}(errChannel)

	for {
		select {
		case err = <-errChannel:
			return err
		case stats := <-statsChannel:
			renderStats := stats.Engines["Render/3D/0"]
			videoStats := stats.Engines["Video/0"]

			metricsctx.Publish(ctx, metricKeyQuicksyncGPURenderUsage, time.Now(), metricsctx.UnitPercent, renderStats.Busy, nil)
			metricsctx.Publish(ctx, metricKeyQuicksyncGPUVideoUsage, time.Now(), metricsctx.UnitPercent, videoStats.Busy, nil)
		}
	}
}

// GPUUsageCUDA monitors the overall, encoder and decoder gpu usages on a CUDA device
func GPUUsageCUDA(ctx context.Context, interval time.Duration) error {
	cmdCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cmd := exec.CommandContext(cmdCtx, "nvidia-smi", "stats")
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to set up stdout pipe for nvidia-smi: %w", err)
	}

	defer func() {
		_ = stdout.Close()
	}()

	go func() {
		err := cmd.Run()
		log.Printf("nvidia-smi exited with: %v", err)
	}()

	recordCh := make(chan []string)
	defer close(recordCh)
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		values := make(map[string][]int)
		for {
			select {
			case record := <-recordCh:
				if len(record) < 4 {
					log.Printf("record too short")
					// shouldn't happen but skip these just in case
					continue
				}

				stat := strings.TrimSpace(record[1])
				valStr := strings.TrimSpace(record[3])

				val, err := strconv.Atoi(valStr)
				if err != nil {
					log.Printf("failed to parse nvidia-smi value: %v", err)
					// just ignore parse errors for now
					continue
				}

				values[stat] = append(values[stat], val)
			case <-ticker.C:
				encUtil := avgInts(values["encUtil"])
				decUtil := avgInts(values["decUtil"])
				gpuUtil := avgInts(values["gpuUtil"])

				metricsctx.Publish(ctx, metricKeyCUDAGPUUsage, time.Now(), metricsctx.UnitPercent, gpuUtil, nil)
				metricsctx.Publish(ctx, metricKeyCUDAGPUEncoderUsage, time.Now(), metricsctx.UnitPercent, encUtil, nil)
				metricsctx.Publish(ctx, metricKeyCUDAGPUDecoderUsage, time.Now(), metricsctx.UnitPercent, decUtil, nil)

				values = make(map[string][]int)
			}
		}
	}()

	reader := csv.NewReader(stdout)
	for {
		record, err := reader.Read()
		if err != nil {
			return fmt.Errorf("failed to read nvidia-smi output: %w", err)
		}

		recordCh <- record
	}
}

func avgInts(vals []int) float64 {
	if len(vals) == 0 {
		return 0
	}

	var sum float64
	for _, v := range vals {
		sum += float64(v)
	}
	return sum / float64(len(vals))
}
