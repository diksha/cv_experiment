// Package server with config info
package server

import "database/sql"

// Server defaults
type Server struct {
	User         string `default:"voxelapp"`
	Password     string `default:"voxelvoxel"`
	DatabaseName string `default:"voxeldev"`
	SslMode      string `default:"disable"`
	Port         int    `default:"50051"`
	Bucket       string `default:"voxel-development-polygon-graph-configs" usage:"the polygon bucket"`
	HealthPort   int    `default:"8081" usage:"port for healthcheck endpoint"`
	ServicePort  int    `default:"8080" usage:"port for service to listen on"`
	PodIP        string `default:"" usage:"ip for the main service to listen on"`
	Environment  string `default:"development" usage:"application environment"`
}

// Config for the server
type Config struct {
	Server
	Database *sql.DB
}
