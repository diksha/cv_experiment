package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/lambda"
)

type Result struct {
	DurationSeconds float64
}

var memInitial = flag.Int("starting-mem", 512, "how much memory for the first test to start with")
var memIncrement = flag.Int("mem-increment", 512, "how much to incrememt memory between runs")
var functionName = flag.String("function-name", "jorge-lambda-transcoder-test", "function name to test against")

func setFunctionMemory(ctx context.Context, client *lambda.Client, functionName string, memory int) error {
	_, err := client.UpdateFunctionConfiguration(ctx, &lambda.UpdateFunctionConfigurationInput{
		FunctionName: aws.String(functionName),
		MemorySize:   aws.Int32(int32(memory)),
	})

	if err != nil {
		return fmt.Errorf("failed to set function memory: %w", err)
	}

	err = lambda.NewFunctionUpdatedV2Waiter(client).Wait(ctx, &lambda.GetFunctionInput{
		FunctionName: aws.String(functionName),
	}, 10*time.Second)
	if err != nil {
		return fmt.Errorf("error while waiting for function update: %w", err)
	}

	return nil
}

func main() {
	log.SetFlags(0)
	flag.Parse()

	ctx := context.Background()
	awsConfig, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		log.Fatal(err)
	}

	lambdaClient := lambda.NewFromConfig(awsConfig)

	mem := *memInitial
	for mem <= 10240 {
		fmt.Printf("Invoking with %d MB...", mem)
		err = setFunctionMemory(ctx, lambdaClient, *functionName, mem)
		if err != nil {
			log.Fatal(err)
		}

		resp, err := lambdaClient.Invoke(ctx, &lambda.InvokeInput{
			FunctionName: functionName,
		})
		if err != nil {
			log.Fatal(err)
		}

		if resp.FunctionError != nil {
			log.Fatal(*resp.FunctionError)
		}

		var res Result
		err = json.Unmarshal(resp.Payload, &res)
		if err != nil {
			log.Fatal(err)
		}

		fmt.Printf("%.2fs\n", res.DurationSeconds)
		mem += *memIncrement
	}
}
