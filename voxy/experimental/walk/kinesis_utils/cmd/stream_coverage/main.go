// Executable to check for gaps in Kinesis Video Streams
package main

import (
	"context"
	"flag"
	"os"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"time"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideo"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideoarchivedmedia"
)

var streamARN = flag.String("stream-arn", "", "ARN of the Kinesis Video Stream to check")
var minGapSize = flag.Duration("min-gap", time.Minute, "Minimum size of a gap to report")
var lookback = flag.Duration("lookback", 168*time.Hour, "How far back to look for gaps")

func main() {
	ctx := context.Background()
	flag.Parse()

	// setup pretty printing logzero console writer
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stdout})

	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		log.Fatal().Err(err).Msg("unable to load SDK config")
	}

	kvClient := kinesisvideo.NewFromConfig(cfg)
	kvamClient := kinesisvideoarchivedmedia.NewFromConfig(cfg)

	streamID := streamIdentiferFromARN(*streamARN)

	kvsClient, err := newKVSClient(ctx, streamID, kvClient, kvamClient)
	if err != nil {
		log.Fatal().Err(err).Msg("unable to create KVS client")
	}

	fragments, err := kvsClient.ListFragments(ctx, time.Now().Add(-*lookback), time.Now())
	if err != nil {
		log.Fatal().Err(err).Msg("unable to list fragments")
	}

	fragSeries := NewFragmentSeries(fragments)

	if fragSeries.IsEmpty() {
		log.Info().Str("stream arn", *streamID.ARN()).Msg("stream has no fragments")
		return
	}

	start := getFragmentStartTime(fragSeries.Fragments[0])
	end := getFragmentEndTime(fragSeries.Fragments[len(fragSeries.Fragments)-1])

	log.Info().
		Str("stream", *streamID.ARN()).
		Int("numFragments", len(fragSeries.Fragments)).
		Time("start", start).
		Time("end", end).
		Msg("Found stream fragment data")

	gaps := fragSeries.CheckContinuity(*minGapSize)

	log.Info().Msgf("Found %d gaps in stream coverage:\n", len(gaps))
	for _, gap := range gaps {
		timeDelta := gap.End.Sub(gap.Start)
		log.Info().Msgf("Coverage Gap of %v: from %v to %v\n", timeDelta, gap.Start, gap.End)
	}
}
