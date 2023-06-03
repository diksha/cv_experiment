package main

import (
	"context"
	"flag"
	"fmt"
	"log"

	consumer "github.com/harlow/kinesis-consumer"
)

func main() {
	var stream = flag.String("stream", "", "Stream name")
	flag.Parse()

	// consumer
	c, err := consumer.New(*stream)
	if err != nil {
		log.Fatalf("consumer error: %v", err)
	}

	// start scan
	err = c.Scan(context.TODO(), func(r *consumer.Record) error {
		fmt.Println(string(r.Data))
		return nil // continue scanning
	})
	if err != nil {
		log.Fatalf("scan error: %v", err)
	}

	// Note: If you need to aggregate based on a specific shard
	// the `ScanShard` function should be used instead.
}
