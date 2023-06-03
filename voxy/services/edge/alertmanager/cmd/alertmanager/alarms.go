package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/cloudwatch"
	"github.com/aws/aws-sdk-go-v2/service/cloudwatch/types"
)

const transcoderFPSAlarmPrefix = "Voxel-Edge-TranscoderFPS"
const edgeHealthAlarmPrefix = "Voxel-Edge-Health"

const breaching = "breaching"
const notBreaching = "notBreaching"
const standardThreshold = 3 * 300 // 3fps * 300s for threshold

func cloudwatchPutMetricAlarm(ctx context.Context, cloudwatchClient *cloudwatch.Client, streamName string, edgeUUID string, threshold float64, treatMissingData string) (*cloudwatch.PutMetricAlarmOutput, error) {
	cloudWatchPutMetricAlarmInput := &cloudwatch.PutMetricAlarmInput{
		AlarmName:          aws.String(getCameraFPSAlarmName(streamName)),
		ComparisonOperator: types.ComparisonOperatorLessThanThreshold,
		EvaluationPeriods:  aws.Int32(1), // 1 period (set to 300s below)
		AlarmDescription:   aws.String(fmt.Sprintf("Camera FPS threshold alarm for %v", streamName)),
		AlarmActions: []string{
			"arn:aws:sns:us-west-2:360054435465:Voxel-Edge-TranscoderFPS-Alarms",
		},
		OKActions: []string{},
		Dimensions: []types.Dimension{{
			Name:  aws.String("EdgeUUID"),
			Value: aws.String(edgeUUID),
		}, {
			Name:  aws.String("StreamName"),
			Value: aws.String(streamName),
		}},
		MetricName: aws.String("FrameTranscoded"),
		Namespace:  aws.String("voxel/edge/transcoder"),
		Period:     aws.Int32(300), // 300s period, streams must be up for 300s to be marked as ok
		Tags: []types.Tag{{
			Key:   aws.String("EdgeUUID"),
			Value: aws.String(edgeUUID),
		}, {
			Key:   aws.String("StreamName"),
			Value: aws.String(streamName),
		}},
		Threshold:        aws.Float64(threshold),
		TreatMissingData: aws.String(treatMissingData),
		Statistic:        types.StatisticSum,
	}

	return cloudwatchClient.PutMetricAlarm(ctx, cloudWatchPutMetricAlarmInput)
}

func getCameraFPSAlarmName(streamName string) string {
	return fmt.Sprintf("%s-%s", transcoderFPSAlarmPrefix, streamName)
}

func getEdgeHealthAlarmName(edgeName string) string {
	return fmt.Sprintf("%s-%s", edgeHealthAlarmPrefix, edgeName)
}

func putCameraFPSAlarm(ctx context.Context, awsConfig aws.Config, edgeUUID, streamName string) error {
	cw := cloudwatch.NewFromConfig(awsConfig)
	_, err := cloudwatchPutMetricAlarm(ctx, cw, streamName, edgeUUID, standardThreshold, breaching)
	if err != nil {
		return fmt.Errorf("PutMetricAlarm request failed: %w", err)
	}
	return nil
}

func getCameraHealthAlarms(ctx context.Context, awsConfig aws.Config) ([]types.MetricAlarm, error) {
	cw := cloudwatch.NewFromConfig(awsConfig)

	var alarms []types.MetricAlarm
	var nextToken *string
	for {
		resp, err := cw.DescribeAlarms(ctx, &cloudwatch.DescribeAlarmsInput{
			AlarmNamePrefix: aws.String(transcoderFPSAlarmPrefix),
			NextToken:       nextToken,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to fetch cloudwatch alarms: %w", err)
		}
		alarms = append(alarms, resp.MetricAlarms...)
		if resp.NextToken == nil {
			break
		}
		nextToken = resp.NextToken
	}

	return alarms, nil
}

func deleteAlarm(ctx context.Context, awsConfig aws.Config, alarmName string) error {
	cw := cloudwatch.NewFromConfig(awsConfig)
	_, err := cw.DeleteAlarms(ctx, &cloudwatch.DeleteAlarmsInput{
		AlarmNames: []string{alarmName},
	})
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	return nil
}

func disableAlarmActions(ctx context.Context, awsConfig aws.Config, alarmName string, edgeUUID string, streamName string) error {
	cw := cloudwatch.NewFromConfig(awsConfig)

	_, err := cloudwatchPutMetricAlarm(ctx, cw, streamName, edgeUUID, -1, notBreaching)
	if err != nil {
		return fmt.Errorf("failed to PutMetricAlarm: %w", err)
	}

	_, err = cw.DisableAlarmActions(ctx, &cloudwatch.DisableAlarmActionsInput{
		AlarmNames: []string{alarmName},
	})
	if err != nil {
		return fmt.Errorf("failed to DisableAlarmActions: %w", err)
	}
	return nil
}

func putEdgeAlarm(ctx context.Context, awsConfig aws.Config, edgeUUID string, alarms []types.MetricAlarm) error {
	cw := cloudwatch.NewFromConfig(awsConfig)
	alarmName := getEdgeHealthAlarmName(edgeUUID)
	alarmRuleEntries := []string{}
	for _, alarm := range alarms {
		alarmRuleEntries = append(alarmRuleEntries, fmt.Sprintf("ALARM(%s)", aws.ToString(alarm.AlarmName)))
	}

	alarmRule := aws.String("FALSE")
	if len(alarms) > 0 {
		alarmRule = aws.String(strings.Join(alarmRuleEntries, " OR "))
	}

	_, err := cw.PutCompositeAlarm(ctx, &cloudwatch.PutCompositeAlarmInput{
		AlarmName:        aws.String(alarmName),
		AlarmRule:        alarmRule,
		AlarmDescription: aws.String(fmt.Sprintf("Edge Health for EdgeUUID=%s", edgeUUID)),
		AlarmActions: []string{
			"arn:aws:sns:us-west-2:360054435465:Voxel-Edge-Health-Alarms",
		},
		OKActions: []string{},
	})
	if err != nil {
		return fmt.Errorf("failed to update alarm %q: %w", alarmName, err)
	}
	return nil
}

func getEdgeAlarms(ctx context.Context, awsConfig aws.Config) ([]types.CompositeAlarm, error) {
	cw := cloudwatch.NewFromConfig(awsConfig)
	resp, err := cw.DescribeAlarms(ctx, &cloudwatch.DescribeAlarmsInput{
		AlarmNamePrefix: aws.String(edgeHealthAlarmPrefix),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get edge alarms: %w", err)
	}

	return resp.CompositeAlarms, nil
}
