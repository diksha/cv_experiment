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
}

// Config for the server
type Config struct {
	Server
	Database *sql.DB
}
