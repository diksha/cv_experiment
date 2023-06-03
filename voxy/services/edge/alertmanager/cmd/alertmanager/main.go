package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/cloudwatch/types"
	"github.com/urfave/cli/v2"
)

func doCheckAndFix(cCtx *cli.Context, fix bool) error {
	ctx := cCtx.Context
	awsConfig := getAWSConfig(cCtx)
	streamFromContext := cCtx.String("stream")
	edge := cCtx.String("edge")

	streams := []string{}
	var err error
	if edge != "" {
		streams, err = determineStreamsInEdge(cCtx)
		if err != nil {
			return fmt.Errorf("failed to list which streams are in the edge: %w", err)
		}
		if len(streams) == 0 {
			return fmt.Errorf("target edge has no streams")
		}
		err = doBatchDisable(cCtx, streams)
		if err != nil {
			return fmt.Errorf("failed to apply the disabled tag to stream: %w", err)
		}
	} else if streamFromContext != "" {
		streams = []string{streamFromContext}
	}

	if err = doCheckAndFixStreams(ctx, awsConfig, fix, streams); err != nil {
		return err
	}

	if streamFromContext != "" {
		edge, err = getEdgeForStream(ctx, awsConfig, streamFromContext)
		if err != nil {
			return fmt.Errorf("failed to get edge for stream %q: %w", streamFromContext, err)
		}
	}
	return doCheckAndFixEdges(cCtx, fix, edge)
}

func doCheckAndFixStreams(ctx context.Context, awsConfig aws.Config, fix bool, streams []string) error {
	if len(streams) == 0 {
		var err error
		streams, err = getKinesisVideoStreams(ctx, awsConfig)
		if err != nil {
			return fmt.Errorf("failed to get kinesis video stream names: %w", err)
		}
	}

	alarms, err := getCameraHealthAlarms(ctx, awsConfig)
	if err != nil {
		return fmt.Errorf("failed to get camera health alarms: %w", err)
	}

	existingAlarms := make(map[string]types.MetricAlarm)
	for _, alarm := range alarms {
		existingAlarms[aws.ToString(alarm.AlarmName)] = alarm
	}

	edges := make(map[string][]string)
	for _, stream := range streams {
		tags, err := getTagsForStream(ctx, awsConfig, stream)
		if err != nil {
			return fmt.Errorf("failed to get tags for stream %q: %w", stream, err)
		}
		edge, ok := tags["edge:allowed-uuid:primary"]
		if !ok {
			// skip streams with no edge config
			continue
		}

		alarmDisabled := tags["alarm"] == "disabled"
		alarmName := getCameraFPSAlarmName(stream)
		edges[edge] = append(edges[edge], stream)
		if fix && alarmDisabled {
			if _, ok := existingAlarms[alarmName]; ok {
				if err := disableAlarmActions(ctx, awsConfig, alarmName, edge, stream); err != nil {
					return fmt.Errorf("failed to disable alarm actions %q: %w", alarmName, err)
				}
			}
			log.Printf("DISABLE: %q", alarmName)
		} else if fix {
			err := putCameraFPSAlarm(ctx, awsConfig, edge, stream)
			if err != nil {
				return fmt.Errorf("error putting metric alarm for %q: %w", stream, err)
			}
			log.Printf("UPDATE: %q", alarmName)
		} else if _, ok := existingAlarms[alarmName]; ok {
			log.Printf("FOUND: %q", alarmName)
		} else if !alarmDisabled {
			log.Printf("MISSING: %q", alarmName)
		} else {
			log.Printf("DISABLED: %q", alarmName)
		}
	}

	return nil
}

func doCheckAndFixEdges(cCtx *cli.Context, fix bool, edgeToCheck string) error {
	ctx := cCtx.Context
	awsConfig := getAWSConfig(cCtx)
	alarms, err := getCameraHealthAlarms(ctx, awsConfig)
	if err != nil {
		return fmt.Errorf("failed to get camera alarms: %w", err)
	}

	edges := make(map[string][]types.MetricAlarm)
	for _, alarm := range alarms {
		for _, dimension := range alarm.Dimensions {
			if aws.ToString(dimension.Name) == "EdgeUUID" {
				edge := aws.ToString(dimension.Value)
				edges[edge] = append(edges[edge], alarm)
				break
			}
		}
	}

	edgeAlarms, err := getEdgeAlarms(ctx, awsConfig)
	if err != nil {
		return fmt.Errorf("failed to get edge alarms: %w", err)
	}

	edgeAlarmSet := make(map[string]bool)
	for _, alarm := range edgeAlarms {
		edgeAlarmSet[aws.ToString(alarm.AlarmName)] = true
	}

	for edge, alarms := range edges {
		if edgeToCheck != "" && edgeToCheck != edge {
			// skip all other edges when edgeToCheck is specified
			continue
		}

		enabledAlarms := []types.MetricAlarm{}
		for _, alarm := range alarms {
			if aws.ToBool(alarm.ActionsEnabled) {
				enabledAlarms = append(enabledAlarms, alarm)
			}
		}

		alarmName := getEdgeHealthAlarmName(edge)

		if fix {
			if err := putEdgeAlarm(ctx, awsConfig, edge, enabledAlarms); err != nil {
				return fmt.Errorf("failed to update edge alarm for %q: %w", edge, err)
			}
			log.Printf("UPDATE: %s", alarmName)
		} else {
			if edgeAlarmSet[alarmName] {
				log.Printf("FOUND: %s", alarmName)
			} else {
				log.Printf("MISSING: %s", alarmName)
			}
		}

		// delete alarms that have been checked from the set of edge alarms
		delete(edgeAlarmSet, alarmName)
	}

	// check through the list of excess alarms that should probably be deleted
	for alarmName := range edgeAlarmSet {
		if edgeToCheck != "" && alarmName != getEdgeHealthAlarmName(edgeToCheck) {
			continue
		}

		if fix {
			if err := deleteAlarm(ctx, awsConfig, alarmName); err != nil {
				return fmt.Errorf("failed to delete alarm %q: %w", alarmName, err)
			}
			log.Printf("DELETED: %s", alarmName)
		} else {
			log.Printf("INVALID: %s", alarmName)
		}
	}

	return nil
}

func doCheck(cCtx *cli.Context) error {
	return doCheckAndFix(cCtx, false)
}

func doFix(cCtx *cli.Context) error {
	return doCheckAndFix(cCtx, true)
}

func doEnable(cCtx *cli.Context) error {
	ctx := cCtx.Context
	awsConfig := getAWSConfig(cCtx)
	streamName := cCtx.String("stream")
	if streamName == "" {
		return fmt.Errorf("stream must be specified")
	}

	if err := setTagsForStream(ctx, awsConfig, streamName, map[string]string{"alarm": "enabled"}); err != nil {
		return fmt.Errorf("failed to set tags for stream %q: %w", streamName, err)
	}

	return doCheckAndFix(cCtx, true)
}

func doBatchDisable(cCtx *cli.Context, streams []string) error {
	ctx := cCtx.Context
	awsConfig := getAWSConfig(cCtx)
	for _, stream := range streams {
		if err := setTagsForStream(ctx, awsConfig, stream, map[string]string{"alarm": "disabled"}); err != nil {
			return fmt.Errorf("failed to set tags for stream %q: %w", stream, err)
		}
	}
	return nil
}

func determineStreamsInEdge(cCtx *cli.Context) ([]string, error) {
	awsConfig := getAWSConfig(cCtx)
	edgeName := cCtx.String("edge")
	ctx := cCtx.Context
	streams, err := getKinesisVideoStreams(ctx, awsConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to get KinesisVideoStreams: %w", err)
	}

	streamsInEdge := []string{}
	for _, stream := range streams {
		tags, err := getTagsForStream(ctx, awsConfig, stream)
		if err != nil {
			return nil, fmt.Errorf("failed to get edge for stream %q: %w", stream, err)

		}
		if tags["edge:allowed-uuid:primary"] == edgeName {
			streamsInEdge = append(streamsInEdge, stream)
		}
	}
	return streamsInEdge, nil
}

func doDisableEdge(cCtx *cli.Context) error {
	edgeName := cCtx.String("edge")
	if edgeName == "" {
		return fmt.Errorf("edge must be specified")
	}

	return doCheckAndFix(cCtx, true)
}

func doDisableStream(cCtx *cli.Context) error {
	ctx := cCtx.Context
	awsConfig := getAWSConfig(cCtx)
	streamName := cCtx.String("stream")
	if streamName == "" {
		return fmt.Errorf("stream must be specified")
	}

	if err := setTagsForStream(ctx, awsConfig, streamName, map[string]string{"alarm": "disabled"}); err != nil {
		return fmt.Errorf("failed to set tags for stream %q: %w", streamName, err)
	}

	return doCheckAndFix(cCtx, true)
}

func doDisable(cCtx *cli.Context) error {
	edgeName := cCtx.String("edge")
	streamName := cCtx.String("stream")
	if streamName != "" && edgeName == "" {
		return doDisableStream(cCtx)
	}
	if streamName == "" && edgeName != "" {
		return doDisableEdge(cCtx)
	}
	if streamName == "" && edgeName == "" {
		return fmt.Errorf("either stream or edge must be specified")
	}
	return fmt.Errorf("both stream and edge are specified, please choose one and run again")
}

func getAWSConfig(cCtx *cli.Context) aws.Config {
	return cCtx.App.Metadata["awsConfig"].(aws.Config)
}

func doStatus(cCtx *cli.Context) error {
	ctx := cCtx.Context
	awsConfig := getAWSConfig(cCtx)

	alarms, err := getCameraHealthAlarms(ctx, awsConfig)
	if err != nil {
		return fmt.Errorf("failed to get alarm status: %w", err)
	}

	okCount := 0
	for _, alarm := range alarms {
		if alarm.StateValue == types.StateValueOk {
			okCount++
		} else {
			log.Printf("%s: %s -- %s", alarm.StateValue, aws.ToString(alarm.AlarmName), aws.ToString(alarm.StateReason))
		}
	}
	log.Println()
	log.Printf("%d NOT OK", len(alarms)-okCount)
	log.Printf("%d OK", okCount)
	return nil
}

func main() {
	log.SetFlags(0)

	ctx := context.Background()
	awsConfig, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		log.Fatalf("failed to load aws config: %v", err)
	}

	flags := []cli.Flag{&cli.StringFlag{
		Name:    "stream",
		Usage:   "operate on a specific stream",
		Aliases: []string{"s"},
	},
		&cli.StringFlag{
			Name:    "edge",
			Usage:   "operate on a specific edge, disabling all streams for that edge",
			Aliases: []string{"e"},
		}}

	app := &cli.App{
		Name:  "alertmanager",
		Usage: "manages camera health alerts",
		Metadata: map[string]interface{}{
			"awsConfig": awsConfig,
		},
		Commands: []*cli.Command{{
			Name:   "check",
			Usage:  "check alert configurations",
			Action: doCheck,
			Flags:  flags,
		}, {
			Name:   "fix",
			Usage:  "fix alert configurations",
			Action: doFix,
			Flags:  flags,
		}, {
			Name:   "enable",
			Usage:  "enable alert for a specific entity",
			Action: doEnable,
			Flags:  flags,
		}, {
			Name:   "disable",
			Usage:  "disable alert for a specific entity",
			Action: doDisable,
			Flags:  flags,
		}, {
			Name:   "status",
			Usage:  "checks current alarm status for all active alarms",
			Action: doStatus,
			Flags:  flags,
		}},
	}

	if err := app.Run(os.Args); err != nil {
		log.Fatal(err)
	}
}
