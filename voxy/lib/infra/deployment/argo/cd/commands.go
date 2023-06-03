//Copyright 2023 Voxel Labs, Inc.
//All rights reserved.
//
//This document may not be reproduced, republished, distributed, transmitted,
//displayed, broadcast or otherwise exploited in any manner without the express
//prior written permission of Voxel Labs, Inc. The receipt or possession of this
//document does not convey any rights to reproduce, disclose, or distribute its
//contents, or to manufacture, use, or sell anything that it may describe, in
//whole or in part.

// Package cd provides the functionality for argocd commands
package cd

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"time"
)

var _ CommandsInterface = (*Commands)(nil)

// CommandsInterface is the interface for the argocd commands
type CommandsInterface interface {
	ClientLogin(ctx context.Context) (string, error)
	ListProjects(ctx context.Context) (string, error)
	ForceCreateApplication(ctx context.Context, deploymentFile string) (string, error)
	CreateApplication(ctx context.Context, deploymentFile string) (string, error)
	DeleteApplication(ctx context.Context, identifier string) (string, error)
}

// Commands is the struct that implements the CommandsInterface
type Commands struct {
	argoServer string
}

// NewCommands creates a new Commands struct
func NewCommands() *Commands {
	return &Commands{
		argoServer: "argo.voxelplatform.com",
	}
}

// ClientLogin logs into the argocd server
func (ac *Commands) ClientLogin(ctx context.Context) (string, error) {
	output, err := ac.executeCommandsRealTimeOutput(ctx, "login", ac.argoServer, "--sso")
	if err != nil {
		return "", fmt.Errorf("failed to login to argo client %w", err)
	}
	return output, nil
}

// ListProjects lists the projects in the argocd server
func (ac *Commands) ListProjects(ctx context.Context) (string, error) {
	output, err := ac.executeCommands(ctx, "proj", "list")
	if err != nil {
		return "", fmt.Errorf("failed to list to argo projects %w", err)
	}
	return output, nil
}

// ForceCreateApplication creates an application in the argocd server
func (ac *Commands) ForceCreateApplication(ctx context.Context, deploymentFile string) (string, error) {
	output, err := ac.executeCommands(ctx, "app", "create", "--upsert", "-f", deploymentFile)
	if err != nil {
		return "", fmt.Errorf("failed to create to argo applications %w", err)
	}
	return output, nil
}

// CreateApplication creates an application in the argocd server
func (ac *Commands) CreateApplication(ctx context.Context, deploymentFile string) (string, error) {
	output, err := ac.executeCommands(ctx, "app", "create", "-f", deploymentFile)
	if err != nil {
		return "", fmt.Errorf("failed to create to argo applications %w", err)
	}
	return output, nil
}

// DeleteApplication deletes an application in the argocd server
func (ac *Commands) DeleteApplication(ctx context.Context, identifier string) (string, error) {
	output, err := ac.executeCommands(ctx, "app", "delete", identifier)
	if err != nil {
		return "", fmt.Errorf("failed to delete to argo applications %w", err)
	}
	return output, nil
}

func (ac *Commands) executeCommands(ctx context.Context, arg ...string) (string, error) {
	// create a context with a timeout
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()

	cmd := exec.CommandContext(ctx, "argocd", arg...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", fmt.Errorf("failed to get the standard out: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("failed to start the command: %w", err)
	}

	var output bytes.Buffer
	scanner := bufio.NewScanner(stdout)
	scanner.Split(bufio.ScanWords)

	for scanner.Scan() {
		m := scanner.Text()
		output.WriteString(m + " ")
	}

	if err := cmd.Wait(); err != nil {
		return "", fmt.Errorf("failed to wait: %w", err)
	}

	return output.String(), nil
}

func (ac *Commands) executeCommandsRealTimeOutput(ctx context.Context, arg ...string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()

	cmd := exec.CommandContext(ctx, "argocd", arg...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", fmt.Errorf("failed to get the standard out: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("failed to start the command: %w", err)
	}

	var output bytes.Buffer
	scanner := bufio.NewScanner(stdout)
	scanner.Split(bufio.ScanWords)

	for scanner.Scan() {
		m := scanner.Text()
		fmt.Println(m)
		output.WriteString(m + " ")
	}

	if err := cmd.Wait(); err != nil {
		return "", fmt.Errorf("failed to wait: %w", err)
	}

	return output.String(), nil
}
