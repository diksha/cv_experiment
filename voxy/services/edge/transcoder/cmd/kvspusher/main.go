// Package main provides an executable for pushing video data to kinesis video streams
package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideo"
	kinesisvideotypes "github.com/aws/aws-sdk-go-v2/service/kinesisvideo/types"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideomedia"
	kinesisvideomediatypes "github.com/aws/aws-sdk-go-v2/service/kinesisvideomedia/types"
	"goji.io"
	"goji.io/pat"

	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg"
	"github.com/voxel-ai/voxel/services/edge/transcoder/lib/kvspusher"
)

var port = flag.Int("port", 8080, "port to listen for requests on")
var streamName = flag.String("stream-name", "", "kinesis stream name to push to")

var testsrc = flag.Bool("testsrc", false, "use ffmpeg testsrc")
var rtspsrc = flag.String("rtspsrc", "", "rtsp src to use as an input")
var streamSrc = flag.String("kvssrc", "", "kinesis video stream name to use as source")
var readKVSProfile = flag.String("read-kvs-profile", "", "specify an alternate profile name (in your ~/.aws/config) to use for reading when specifying kvssrc")
var debugChunks = flag.Bool("debug-chunks", false, "save debug chunks")
var readKVSStartOffset = flag.Duration("read-kvs-offset", time.Duration(0), "when reading from KVS stream, specify the offset from the current time to fetch from. Accurate to the second")

func logRequests() func(http.Handler) http.Handler {
	return func(h http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			log.Printf("HTTP START %s %s", r.Method, r.URL)
			start := time.Now()
			h.ServeHTTP(w, r)
			log.Printf("HTTP END   %s %s %.2fms", r.Method, r.URL, float64(time.Since(start))/float64(time.Millisecond))
		})
	}
}

func limitBody(size int64) func(http.Handler) http.Handler {
	return func(h http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.ContentLength > size {
				http.Error(w, fmt.Sprintf("invalid content length > %d", size), http.StatusBadRequest)
				return
			}
			h.ServeHTTP(w, r)
		})
	}
}

func handleFragments(client *kvspusher.Client, ch <-chan []byte) {
	debugIndex := 0
	for data := range ch {
		frag, err := kvspusher.ReadFragment(bytes.NewReader(data))
		if err != nil {
			log.Fatalf("failed to read mkv input: %v", err)
		}

		if *debugChunks {
			log.Printf("writing debug chunks")
			err = os.WriteFile(fmt.Sprintf("input%d.mkv", debugIndex), data, 0600)
			if err != nil {
				log.Fatalf("failed to write debug chunk: %v", err)
			}

			var buf bytes.Buffer
			err = kvspusher.WriteFragment(&buf, kvspusher.SimplifyFragment(frag))
			if err != nil {
				log.Fatalf("failed to marshal output chunk: %v", err)
			}

			err = os.WriteFile(fmt.Sprintf("output%d.mkv", debugIndex), buf.Bytes(), 0600)
			if err != nil {
				log.Fatalf("failed to write output chunk: %v", err)
			}
			debugIndex++
		}

		err = func() error {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			// trunk-ignore(golangci-lint/wrapcheck)
			return client.PutMediaFragment(ctx, frag)
		}()
		if err != nil {
			log.Printf("KVS UPLOAD ERROR %s", err)
		}

		log.Printf("KVS UPLOAD COMPLETE LEN=%d", len(data))
	}
}

type sourceType string

const (
	sourceTypeNone = sourceType("")
	sourceTypeTest = sourceType("test")
	sourceTypeRTSP = sourceType("rtsp")
	sourceTypeKVS  = sourceType("kvs")
)

func getSourceType() sourceType {
	numSrcsDefined := 0
	sourceType := sourceTypeNone

	if *testsrc {
		sourceType = sourceTypeTest
		numSrcsDefined++
	}

	if *rtspsrc != "" {
		sourceType = sourceTypeRTSP
		numSrcsDefined++
	}

	if *streamSrc != "" {
		sourceType = sourceTypeKVS
		numSrcsDefined++
	}

	if numSrcsDefined != 1 {
		log.Fatal("must specify exactly one of -testsrc, -rtspsrc and -stream-src")
	}

	return sourceType
}

func getCommandArgs(srcType sourceType, httpPort int) (allFlags []string) {
	globalFlags := []string{
		"-y",
		"-hide_banner",
		"-loglevel", "error",
		"-nostdin",
	}

	if srcType == sourceTypeTest || srcType == sourceTypeRTSP {
		globalFlags = append(globalFlags, "-re")
	}

	allFlags = append(allFlags, globalFlags...)

	outputFlags := []string{
		"-vf", "setpts=(PTS-STARTPTS)+RTCSTART/(TB*1000000)",
		"-vsync", "0",
		"-force_key_frames", "expr:if(isnan(prev_forced_t),1,gte(t,prev_forced_t+8))",
		"-preset", "fast",
		"-x265-params", "aud=1",
		"-x265-params", "no-info=1",
		"-c:v", "libx265",
		"-bsf:v", "hevc_metadata=aud=insert",
		"-f", "segment",
		fmt.Sprintf("http://localhost:%d", httpPort) + "/segment/chunk%d.mkv",
	}
	allFlags = append(allFlags, outputFlags...)

	// Set input flags
	switch srcType {
	case sourceTypeTest:
		allFlags = append(allFlags,
			"-f", "lavfi",
			"-i", "testsrc=d=6000:s=1280x720:r=5,format=yuv420p",
		)
	case sourceTypeRTSP:
		allFlags = append(allFlags,
			"-f", "rtsp",
			"-i", *rtspsrc,
		)
	case sourceTypeKVS:
		allFlags = append(allFlags,
			"-i", "-",
		)
	default:
		log.Fatalf("unsupported source type %q", srcType)
	}

	return allFlags
}

func doFetchFromKVS(ctx context.Context, kvsClient *kinesisvideomedia.Client, streamName *string, endpointResolver kinesisvideomedia.EndpointResolver, payloadWriter io.Writer) int64 {
	var input kinesisvideomedia.GetMediaInput

	if readKVSStartOffset.Truncate(time.Second).Seconds() == 0 {
		input = kinesisvideomedia.GetMediaInput{
			StreamName: streamName,
			StartSelector: &kinesisvideomediatypes.StartSelector{
				StartSelectorType: kinesisvideomediatypes.StartSelectorTypeNow,
			},
		}
	} else {
		timestamp := time.Now().Add(-*readKVSStartOffset)
		input = kinesisvideomedia.GetMediaInput{
			StreamName: streamName,
			StartSelector: &kinesisvideomediatypes.StartSelector{
				StartSelectorType: kinesisvideomediatypes.StartSelectorTypeServerTimestamp,
				StartTimestamp:    &timestamp,
			},
		}
	}

	log.Printf("Calling GetMedia")
	getMediaResult, err := kvsClient.GetMedia(ctx, &input, kinesisvideomedia.WithEndpointResolver(endpointResolver))
	if err != nil {
		log.Fatalf("failed to get media: %v", err)
	}
	defer func() {
		_ = getMediaResult.Payload.Close()
	}()

	contentType := aws.ToString(getMediaResult.ContentType)
	if contentType == "application/json" {
		contentBytes, err := io.ReadAll(getMediaResult.Payload)
		errMsg := "GetMedia returned JSON payload"
		if err != nil {
			errMsg += ": " + string(contentBytes)
		}
		log.Fatal(errMsg)
	} else if contentType != "video/webm" {
		log.Fatalf("GetMedia returned payload with unsupported content type %q", contentType)
	}

	bytesWritten, err := io.Copy(payloadWriter, getMediaResult.Payload)
	if err != nil {
		log.Fatalf("failed copy from GetMedia payload to pipe: %v", err)
	}

	log.Printf("copy from media payload exited with %v bytes written", bytesWritten)
	return bytesWritten
}

func fetchFromKVS(ctx context.Context, readStreamName, awsProfileName string, output io.WriteCloser) {
	defer func() {
		_ = output.Close()
	}()

	cfgOptions := []func(*config.LoadOptions) error{}
	if awsProfileName != "" {
		cfgOptions = append(cfgOptions, config.WithSharedConfigProfile(awsProfileName))
	}

	cfg, err := config.LoadDefaultConfig(
		ctx,
		cfgOptions...,
	)
	if err != nil {
		log.Fatal(err)
	}

	kinesisClient := kinesisvideo.NewFromConfig(cfg)
	endpointOutput, err := kinesisClient.GetDataEndpoint(
		ctx,
		&kinesisvideo.GetDataEndpointInput{
			APIName:    kinesisvideotypes.APINameGetMedia,
			StreamName: aws.String(readStreamName),
		},
	)
	if err != nil {
		log.Fatalf("failed call to GetDataEndpoint: %v", err)
	}
	endpointResolver := kinesisvideomedia.EndpointResolverFromURL(
		aws.ToString(endpointOutput.DataEndpoint),
	)

	pullClient := kinesisvideomedia.NewFromConfig(cfg)
	noDataStreak := 0
	for {
		bytesWritten := doFetchFromKVS(ctx, pullClient, aws.String(readStreamName), endpointResolver, output)
		if bytesWritten == 0 {
			noDataStreak++
			backoffSeconds := int64(math.Pow(2, float64(noDataStreak)))
			log.Printf("KVS fetcher sleeping for %v seconds", backoffSeconds)
			time.Sleep(time.Duration(backoffSeconds) * time.Second)
		} else {
			noDataStreak = 0
		}
	}
}

func main() {
	ctx := context.Background()

	flag.Parse()
	srcType := getSourceType()

	fragmentCh := make(chan []byte)

	mux := goji.NewMux()
	mux.Use(logRequests())
	// limit body to 50mb which is the fragment size limit for kinesis video
	mux.Use(limitBody(1024 * 1024 * 50))

	mux.HandleFunc(pat.Post("/segment/:name"), func(w http.ResponseWriter, r *http.Request) {
		// still need this limitreader in case of chunked transfer requests
		data, err := io.ReadAll(io.LimitReader(r.Body, 1024*1024*50))
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to ready request body: %s", err), http.StatusInternalServerError)
			return
		}

		fragmentCh <- data
	})

	// set up the aws kinesis video api client
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion("us-west-2"))
	if err != nil {
		log.Fatal(err)
	}

	pusher, err := kvspusher.Init(ctx, *streamName, cfg)
	if err != nil {
		log.Fatal(err)
	}

	go handleFragments(pusher, fragmentCh)

	listenAddr := net.JoinHostPort("localhost", strconv.Itoa(*port))
	if *testsrc {
		listenAddr = "localhost:0"
	}

	httpLn, err := net.Listen("tcp", listenAddr)
	if err != nil {
		log.Fatal("failed to listen:", err)
	}

	var httpPort int
	if tcpAddr, ok := httpLn.Addr().(*net.TCPAddr); ok {
		httpPort = tcpAddr.Port
	}

	flags := getCommandArgs(srcType, httpPort)
	cmd, err := ffmpeg.New(ctx, flags...)
	if err != nil {
		log.Fatalf("failed to create ffmpeg command: %v", err)
	}

	if srcType == sourceTypeKVS {
		r, w := io.Pipe()
		ffmpegInputPipe, err := cmd.StdinPipe()
		if err != nil {
			log.Fatalf("failed to get ffmpeg stdin pipe: %v", err)
		}

		go fetchFromKVS(ctx, *streamSrc, *readKVSProfile, w)

		go func() {
			_, err = io.Copy(ffmpegInputPipe, r)
			if err != nil {
				log.Fatalf("failed to copy from pipe to ffmpeg stdin: %v", err)
			}
		}()
	}

	go func() {
		// this is a little gross but we're just going to sleep
		// so that the listener starts before ffmpeg does
		time.Sleep(3 * time.Second)

		err := cmd.Start()
		if err != nil {
			log.Fatalf("failed to start ffmpeg command: %v", err)
		}

		go func() {
			commandRunning := true
			for commandRunning {
				select {
				case <-cmd.Done():
					commandRunning = false
				case line := <-cmd.Output():
					log.Println(line)
				}
			}
		}()

		log.Fatal("ffmpeg exited:", cmd.Wait())
	}()

	log.Fatal(http.Serve(httpLn, mux))
}
