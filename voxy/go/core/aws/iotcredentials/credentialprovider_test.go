package iotcredentials_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/go/core/aws/iotcredentials"
)

func mustParseURL(t *testing.T, urlString string) *url.URL {
	u, err := url.Parse(urlString)
	require.NoError(t, err, "url must parse")
	return u
}

func TestCredentialsProvider(t *testing.T) {
	var resp iotcredentials.IOTEndpointOutput
	resp.Credentials.AccessKeyID = "access-key-test"
	resp.Credentials.SecretAccessKey = "secret-key-test"
	resp.Credentials.SessionToken = "session-token-test"
	resp.Credentials.Expiration = time.Now().Format(time.RFC3339)

	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/role-aliases/test-role/credentials", r.URL.Path, "path should be correct match")
		assert.Equal(t, "test-thing", r.Header.Get("x-amzn-iot-thingname"), "thing name header should be correct")

		respBody, err := json.Marshal(resp)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to marshal response: %v", err), http.StatusInternalServerError)
			return
		}

		// trunk-ignore(semgrep/go.lang.security.audit.xss.no-direct-write-to-responsewriter.no-direct-write-to-responsewriter): not helpful here
		_, err = w.Write(respBody)
		assert.NoError(t, err, "should write response")
	}))
	defer srv.Close()

	endpoint := mustParseURL(t, srv.URL).Host
	tlsConfig := srv.Client().Transport.(*http.Transport).TLSClientConfig
	provider, err := iotcredentials.NewProvider(endpoint, "test-thing", "test-role", iotcredentials.WithTLSConfig(tlsConfig))
	require.NoError(t, err, "must load iot credentials provider")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	creds, err := provider.Retrieve(ctx)
	require.NoError(t, err, "should get credentials successfully")

	assert.Equal(t, resp.Credentials.AccessKeyID, creds.AccessKeyID, "access key matches")
	assert.Equal(t, resp.Credentials.SecretAccessKey, creds.SecretAccessKey, "secret key matches")
	assert.Equal(t, resp.Credentials.SessionToken, creds.SessionToken, "sesison token matches")
	assert.Equal(t, resp.Credentials.Expiration, creds.Expires.Format(time.RFC3339), "expiration matches")
	assert.True(t, creds.CanExpire, "can expire set correctly")
}
