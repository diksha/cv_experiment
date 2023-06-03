// Package ingest is the entry point for processing incidents sent to Prism.
package ingest

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aws/aws-lambda-go/events"
	"github.com/davecgh/go-spew/spew"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/voxel-ai/voxel/services/platform/prism/lib/fragarchive"
	"github.com/voxel-ai/voxel/services/platform/prism/lib/incident"
	"github.com/voxel-ai/voxel/services/platform/prism/lib/videoarchiver"
)

// Client is a client for the ingest service.
type Client struct {
	ArchiveClient *fragarchive.Client
	KVAMClient    videoarchiver.KinesisVideoArchivedMediaAPI
	KVClient      videoarchiver.KinesisVideoAPI
}

// HandleSQSEvent is the entrypoint for the ingest service from SQS.
func (c *Client) HandleSQSEvent(ctx context.Context, event events.SQSEvent) error {
	logger := log.Level(zerolog.DebugLevel).With().Logger()
	ctx = logger.WithContext(ctx)

	eventJSON, err := json.Marshal(event)
	if err != nil {
		log.Debug().Err(err).Msg("failed to marshal event")
		log.Debug().
			Int("Num Records", len(event.Records)).
			Msgf("received SQS event")
	} else {
		log.Debug().RawJSON("event", eventJSON).Msg("received SQS event")
	}

	if len(event.Records) != 1 {
		log.Ctx(ctx).Debug().Str("Record Dump", spew.Sdump(event.Records)).Msg("invalid number of records")
		return fmt.Errorf("expected one record but received %v", len(event.Records))
	}

	return c.handleSQSMessage(ctx, &event.Records[0])
}

func (c *Client) handleSQSMessage(ctx context.Context, message *events.SQSMessage) error {
	logger := log.Ctx(ctx).With().Str("MessageID", message.MessageId).Logger()
	ctx = logger.WithContext(ctx)

	type MessageBody struct {
		Message string
	}
	var result MessageBody
	if err := json.Unmarshal([]byte(message.Body), &result); err != nil {
		logger.Debug().Msgf("Message Body: %v", message.Body)
		return fmt.Errorf("failed to unmarshall messsage body: %w", err)
	}

	incident, err := incident.Unmarshal(result.Message)
	if err != nil {
		return fmt.Errorf("failed to unmarshall incident from JSON: %w", err)
	}

	return c.HandleIncident(ctx, incident)
}

// HandleIncident handles an incident to be processed by Prism.
func (c *Client) HandleIncident(ctx context.Context, incident *incident.Incident) error {
	archiver, err := videoarchiver.New(
		ctx,
		videoarchiver.Config{
			FragmentArchiveClient:           c.ArchiveClient,
			KinesisVideoStreamARN:           incident.StreamARN,
			KinesisVideoClient:              c.KVClient,
			KinesisVideoArchivedMediaClient: c.KVAMClient,
			CameraUUID:                      incident.CameraUUID,
		},
	)
	if err != nil {
		return fmt.Errorf("failed to create fragment archiver: %w", err)
	}

	err = archiver.ArchiveTimeRange(
		ctx,
		incident.GetClipStartTime(),
		incident.GetClipEndTime(),
	)
	if err != nil {
		return fmt.Errorf("failed to archive time range: %w", err)
	}

	return nil
}
