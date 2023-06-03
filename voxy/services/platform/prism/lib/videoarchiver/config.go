package videoarchiver

import (
	"errors"

	"github.com/rs/zerolog/log"

	"github.com/voxel-ai/voxel/services/platform/prism/lib/fragarchive"
)

// Config is the configurable input for creating a new fragment archiving client.
// Must specify exactly one of KinesisVideoStreamARN or KinesisVideoStreamName.
type Config struct {
	FragmentArchiveClient           *fragarchive.Client
	KinesisVideoStreamARN           string
	KinesisVideoStreamName          string
	KinesisVideoClient              KinesisVideoAPI
	KinesisVideoArchivedMediaClient KinesisVideoArchivedMediaAPI
	CameraUUID                      string
}

func (config *Config) getStreamIdentifier() (streamID streamIdentifier, err error) {
	arnDefined := config.KinesisVideoStreamARN != ""
	nameDefined := config.KinesisVideoStreamName != ""

	if !arnDefined && !nameDefined {
		err = errors.New("must define KinesisVideoStreamARN or KinesisVideoStreamName")
	} else if !arnDefined {
		streamID = streamIdentiferFromName(config.KinesisVideoStreamName)
	} else {
		streamID = streamIdentiferFromARN(config.KinesisVideoStreamARN)
		if nameDefined {
			log.Warn().Msg("specified both KinesisVideoStreamARN and KinesisVideoStreamName, ignoring name")
		}
	}

	return streamID, err
}
