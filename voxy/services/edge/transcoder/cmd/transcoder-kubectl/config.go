package main

import (
	"context"
	"fmt"
	"io/ioutil"
	"os/exec"

	appsv1 "k8s.io/client-go/applyconfigurations/apps/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/yaml"

	"github.com/voxel-ai/voxel/lib/infra/edge/edgeconfig"
	edgeconfigpb "github.com/voxel-ai/voxel/protos/edge/edgeconfig/v1"
)

func loadKubeYaml(filename string) (*appsv1.StatefulSetApplyConfiguration, error) {
	rawYaml, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read kube yaml %q: %w", filename, err)
	}

	var statefulSet appsv1.StatefulSetApplyConfiguration
	if err := yaml.Unmarshal(rawYaml, &statefulSet); err != nil {
		return nil, fmt.Errorf("failed to parse kube yaml: %w", err)
	}

	return &statefulSet, nil
}

func loadKubeconfigFromMicrok8s(ctx context.Context) ([]byte, error) {
	cmd := exec.CommandContext(ctx, "microk8s", "config")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("failed to get kubeconfig from microk8s")
	}

	return out, nil
}

func loadKubeconfig(ctx context.Context, filename string) (*restclient.Config, error) {
	var configBytes []byte
	var err error

	if filename == "" {
		configBytes, err = loadKubeconfigFromMicrok8s(ctx)
	} else {
		configBytes, err = ioutil.ReadFile(filename)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to load kubeconfig: %w", err)
	}

	return clientcmd.RESTConfigFromKubeConfig(configBytes)
}

// attempt to load an edge config from a file, secrets manager otherwise
func loadEdgeConfig(ctx context.Context, filename string) (*edgeconfigpb.EdgeConfig, error) {
	out, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read %q: %w", filename, err)
	}

	cfg, err := edgeconfig.ParseYAML(string(out))
	if err != nil {
		return nil, fmt.Errorf("failed to parse %q: %w", filename, err)
	}
	return cfg, nil
}
