// Package main for spinning up the gRPC client
package main

import (
	"context"
	"flag"
	"log"
	"time"

	"github.com/davecgh/go-spew/spew"
	_ "github.com/lib/pq"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/voxel-ai/voxel/experimental/jorge/polygonpoc"
)

var (
	addr = flag.String("addr", "localhost:50051", "the address to connect to")
)

func main() {
	conn, err := grpc.Dial(*addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("failed to connect: %v", err)
	}
	defer func() {
		err = conn.Close()
		if err != nil {
			log.Fatalf("could not close the connection: %v", err)
		}
	}()
	c := polygonpoc.NewPolygonServiceClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	r, err := c.GetCameras(ctx, &polygonpoc.GetCamerasRequest{})
	if err != nil {
		log.Fatalf("could not get cameras: %v", err)
	}
	spew.Dump("List of cameras: %s", r.GetCameras())
}
