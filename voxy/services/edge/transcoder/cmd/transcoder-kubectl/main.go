package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log"
	"os/exec"
	"regexp"
	"time"

	"google.golang.org/protobuf/encoding/protojson"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	appsv1apply "k8s.io/client-go/applyconfigurations/apps/v1"
	corev1apply "k8s.io/client-go/applyconfigurations/core/v1"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/yaml"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/cristalhq/aconfig"
	"github.com/cristalhq/aconfig/aconfigyaml"
	"github.com/davecgh/go-spew/spew"
	"google.golang.org/protobuf/proto"

	edgeconfigpb "github.com/voxel-ai/voxel/protos/edge/edgeconfig/v1"
)

// These are the environment variable names consumed by the transcoder. We will
// populate them in the statefulset apply configuration
var (
	iotCredsEndpointEnv = "IOT_CREDS_ENDPOINT"
	iotRoleAliasEnv     = "IOT_ROLE_ALIAS"
	iotThingNameEnv     = "IOT_THING_NAME"
)

// Config specifies the configuration values for this program
type Config struct {
	EdgeConfig string `usage:"edge config file path"`
	DryRun     bool   `usage:"whether this hsould be a dry run or actually generate files/run kubectl apply"`
	Kubeconfig string `usage:"kubeconfig file to load, will attempt to automatically load from microk8s if unset"`

	StatefulSetYAML string `usage:"statefulset yaml file to base our statefuleset on"`
	ImageName       string `usage:"image name for the kubernetes container"`

	IOT struct {
		ThingName     string `usage:"iot thing name to use"`
		CredsEndpoint string `usage:"iot creds endpoint to use"`
		RoleAlias     string `usage:"iot role alias to use"`
	}
}

func calculateConfigHash(edgeCfg *edgeconfigpb.EdgeConfig) (string, error) {
	data, err := proto.Marshal(edgeCfg)
	if err != nil {
		return "", fmt.Errorf("failed to marshal proto string: %w", err)
	}

	hasher := sha256.New()
	_, err = hasher.Write(data)
	if err != nil {
		return "", fmt.Errorf("failed to compute proto message hash: %w", err)
	}

	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func prepareConfigMap(ctx context.Context, podPrefix string, streams []*edgeconfigpb.StreamConfig) (*corev1apply.ConfigMapApplyConfiguration, error) {
	configMap := corev1apply.ConfigMap(fmt.Sprintf("%s-configs", podPrefix), "default")

	binaryData := make(map[string]string)

	for i, streamConfig := range streams {
		data, err := protojson.MarshalOptions{
			Multiline: true,
			Indent:    "    ",
		}.Marshal(streamConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal streamconfig.json: %w", err)
		}
		binaryData[fmt.Sprintf("%s-%d.json", podPrefix, i)] = string(data)
	}

	configMap.WithData(binaryData)

	return configMap, nil
}

func prepareStatefulSet(ctx context.Context, cfg Config, edgeCfg *edgeconfigpb.EdgeConfig, awsConfig aws.Config) (*appsv1apply.StatefulSetApplyConfiguration, error) {
	statefulSet, err := loadKubeYaml(cfg.StatefulSetYAML)
	if err != nil {
		return nil, fmt.Errorf("failed to load statefulset yaml: %w", err)
	}

	// set the number of replicas to equal the number of streams
	statefulSet.Spec.WithReplicas(int32(len(edgeCfg.Streams)))

	// this should always be set
	if statefulSet.ObjectMetaApplyConfiguration == nil {
		return nil, fmt.Errorf("StatefulSet does not have .metadata set")
	}

	// this should always be set
	if statefulSet.ObjectMetaApplyConfiguration.Name == nil {
		return nil, fmt.Errorf("StatefulSet does not have .metadata.name set")
	}

	// find the edge-transcoder entry and apply some dynamic configuration values to it
	foundTranscodeContainer := false
	for i, container := range statefulSet.Spec.Template.Spec.Containers {
		if aws.ToString(container.Image) == "edge-transcoder" {
			foundTranscodeContainer = true
			statefulSet.Spec.Template.Spec.Containers[i].
				WithImage(cfg.ImageName).
				WithEnv(&corev1apply.EnvVarApplyConfiguration{
					Name:  &iotCredsEndpointEnv,
					Value: &cfg.IOT.CredsEndpoint,
				}, &corev1apply.EnvVarApplyConfiguration{
					Name:  &iotRoleAliasEnv,
					Value: &cfg.IOT.RoleAlias,
				}, &corev1apply.EnvVarApplyConfiguration{
					Name:  &iotThingNameEnv,
					Value: &cfg.IOT.ThingName,
				})
		}
	}

	if !foundTranscodeContainer {
		return nil, fmt.Errorf("failed to find container with image name 'edge-transcoder' for replacement in statefulset")
	}

	configHash, err := calculateConfigHash(edgeCfg)
	if err != nil {
		return nil, fmt.Errorf("error while calculating config hash for statefulset: %w", err)
	}

	statefulSet.Spec.Template.WithAnnotations(map[string]string{
		"configHash": configHash,
	})

	return statefulSet, nil
}

const dockerImageVerifyRegex = `^[a-zA-Z0-9\.\-]+(\:\d)?((\/[a-zA-Z0-9\_\-\.]+)+\:[a-zA-Z0-9_\-\.]{1,128})?`

// compile it at init time
var dockerImageVerify = regexp.MustCompile(dockerImageVerifyRegex)

func isValidImageName(img string) bool {
	return dockerImageVerify.FindString(img) != ""
}

func copyImageFromDocker(ctx context.Context, img string) error {
	if !isValidImageName(img) {
		return fmt.Errorf("invalid image tag %q", img)
	}
	cmd := exec.CommandContext(ctx, "/bin/sh", "-ce", fmt.Sprintf("docker save %q | microk8s ctr image import -", img))
	// trunk-ignore(semgrep/trailofbits.go.invalid-usage-of-modified-variable.invalid-usage-of-modified-variable): out is valid even on err
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("microk8s ctr import failed: %v\n\n%s", err, string(out))
	}
	return nil
}

func applyConfigs(ctx context.Context, cfg Config, configMap *corev1apply.ConfigMapApplyConfiguration, statefulSet *appsv1apply.StatefulSetApplyConfiguration) (*corev1.ConfigMap, *appsv1.StatefulSet, error) {
	kubeconfig, err := loadKubeconfig(ctx, cfg.Kubeconfig)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load kubeconfig: %w", err)
	}

	clientset, err := kubernetes.NewForConfig(kubeconfig)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to construct kubernetes client: %w", err)
	}

	if cfg.DryRun {
		configMapBytes, err := yaml.Marshal(configMap)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to marshal configmap yaml: %w", err)
		}

		statefulSetBytes, err := yaml.Marshal(statefulSet)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to marshal statefulset yaml: %w", err)
		}
		log.Println("---------configmap.yaml---------")
		log.Println(string(configMapBytes))
		log.Println("---------------EOF--------------")
		log.Println("--------statefulset.yaml--------")
		log.Println(string(statefulSetBytes))
		log.Println("---------------EOF--------------")
	}

	var dryRunOption []string
	if cfg.DryRun {
		dryRunOption = []string{"All"}
	}

	applyOptions := metav1.ApplyOptions{
		DryRun:       dryRunOption,
		FieldManager: "transcoder-kubectl",
		Force:        true,
	}

	configMapRes, err := clientset.CoreV1().ConfigMaps("default").Apply(ctx, configMap, applyOptions)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to apply configmap kubernetes config: %w", err)
	}

	statefulSetRes, err := clientset.AppsV1().StatefulSets("default").Apply(ctx, statefulSet, applyOptions)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to apply statefulset kubernetes config: %w", err)
	}

	return configMapRes, statefulSetRes, nil
}

// This tool is used to generate a set of environment configs from an input yaml
func main() {
	log.SetFlags(0)

	spew.Config = spew.ConfigState{
		Indent:                  "  ",
		DisablePointerAddresses: true,
		DisableCapacities:       true,
		SortKeys:                true,
	}

	var cfg Config
	loader := aconfig.LoaderFor(&cfg, aconfig.Config{
		FileDecoders: map[string]aconfig.FileDecoder{
			".yaml": aconfigyaml.New(),
		},
		FileFlag: "config",
	})

	if err := loader.Load(); err != nil {
		log.Fatal(err)
	}

	log.Printf("loaded config:\n%s", spew.Sdump(cfg))

	log.Printf("copying image %s from docker", cfg.ImageName)
	if !cfg.DryRun {
		err := copyImageFromDocker(context.Background(), cfg.ImageName)
		if err != nil {
			log.Fatalf("failed to copy image %s from docker: %v", cfg.ImageName, err)
		}
	}

	awsConfig, err := config.LoadDefaultConfig(context.Background(), config.WithRegion("us-west-2"))
	if err != nil {
		log.Fatalf("failed to load aws config: %v", err)
	}

	edgeCfg, err := loadEdgeConfig(context.Background(), cfg.EdgeConfig)
	if err != nil {
		log.Fatalf("failed to load edge config: %v", err)
	}

	log.Printf("loaded edge config:\n%s", protojson.Format(edgeCfg))

	statefulSet, err := prepareStatefulSet(context.Background(), cfg, edgeCfg, awsConfig)
	if err != nil {
		log.Fatalf("failed to prepare StatefulSet: %v", err)
	}

	configMap, err := prepareConfigMap(context.Background(), *statefulSet.ObjectMetaApplyConfiguration.Name, edgeCfg.GetStreams())
	if err != nil {
		log.Fatalf("failed to prepare ConfigMap: %v", err)
	}

	applyCtx, applyCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer applyCancel()
	configMapRes, statefulSetRes, err := applyConfigs(applyCtx, cfg, configMap, statefulSet)
	if err != nil {
		log.Fatalf("failed to apply StatefulSet: %v", err)
	}

	configMapOut, err := yaml.Marshal(configMapRes)
	if err != nil {
		log.Fatalf("failed to marshal kubernetes response: %v", err)
	}

	log.Println("Successful response from kubernetes, responsed data:")
	log.Print(string(configMapOut))

	statefulSetOut, err := yaml.Marshal(statefulSetRes)
	if err != nil {
		log.Fatalf("failed to marshal kubernetes response: %v", err)
	}

	log.Println("Successful response from kubernetes, responsed data:")
	log.Print(string(statefulSetOut))
}
