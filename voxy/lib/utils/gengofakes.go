package util

import "net/http"

// This package generates fake implementations of common interfaces for testing purposes

// HTTPClient is the interface commonly used by consumers of http.Client
//
//go:generate go run github.com/maxbrunsfeld/counterfeiter/v6 -o gofakes . HTTPClient
type HTTPClient interface {
	Do(*http.Request) (*http.Response, error)
}
