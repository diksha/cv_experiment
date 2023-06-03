package main

import (
	"fmt"

	"github.com/aws/aws-lambda-go/events"
	"github.com/aws/aws-lambda-go/lambda"
)

func handleRequest(evnt events.KinesisFirehoseEvent) (events.KinesisFirehoseResponse, error) {
	fmt.Printf("InvocationID: %s\n", evnt.InvocationID)
	fmt.Printf("DeliveryStreamArn: %s\n", evnt.DeliveryStreamArn)
	fmt.Printf("Region: %s\n", evnt.Region)

	var response events.KinesisFirehoseResponse

	for _, record := range evnt.Records {
		var transformedRecord events.KinesisFirehoseResponseRecord
		transformedRecord.RecordID = record.RecordID
		transformedRecord.Result = events.KinesisFirehoseTransformedStateOk
		transformedRecord.Data = record.Data

		var metaData events.KinesisFirehoseResponseRecordMetadata
		partitionKeys := make(map[string]string)

		partitionKeys["camera_uuid"] = record.KinesisFirehoseRecordMetadata.PartitionKey
		metaData.PartitionKeys = partitionKeys
		transformedRecord.Metadata = metaData

		response.Records = append(response.Records, transformedRecord)
	}

	fmt.Printf("Processed %d records\n", len(evnt.Records))

	return response, nil
}

func main() {
	lambda.Start(handleRequest)
}
