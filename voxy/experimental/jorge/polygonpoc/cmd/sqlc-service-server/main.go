package main

import (
	"database/sql"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	"github.com/cristalhq/aconfig"
	"google.golang.org/grpc"

	"github.com/voxel-ai/voxel/experimental/jorge/polygonpoc"
	polygonserver "github.com/voxel-ai/voxel/experimental/jorge/polygonpoc/lib/server"
)

func main() {
	var cfg polygonserver.Config
	loader := aconfig.LoaderFor(&cfg, aconfig.Config{EnvPrefix: "POLYGON"})
	if err := loader.Load(); err != nil {
		log.Fatalf("failed to load config: %v", err)
	}
	db, err := sql.Open("postgres", fmt.Sprintf("user=%s password=%s dbname=%s sslmode=%s", cfg.User, cfg.Password, cfg.DatabaseName, cfg.SslMode))
	if err != nil {
		log.Fatalf("failed to open db connection: %v", err)
	}
	cfg.Database = db

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.Port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	gRPCServer := grpc.NewServer()

	idleConnsClosed := make(chan struct{})
	go func() {
		sigint := make(chan os.Signal, 2)
		signal.Notify(sigint, os.Interrupt)
		signal.Notify(sigint, syscall.SIGTERM)
		<-sigint

		go func() {
			<-sigint
			log.Fatal("sigterm received during shutdown attempt")
		}()

		gRPCServer.GracefulStop()
		close(idleConnsClosed)
	}()

	polygonpoc.RegisterPolygonServiceServer(gRPCServer, polygonserver.NewServer(cfg))
	log.Printf("server listening at %v", lis.Addr())
	if err := gRPCServer.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
