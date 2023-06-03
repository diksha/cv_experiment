package main

import (
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"strconv"

	"github.com/cristalhq/aconfig"
)

type config struct {
	Port int `default:"8080" usage:"port to listen on"`
}

func main() {
	var cfg config
	loader := aconfig.LoaderFor(&cfg, aconfig.Config{})
	if err := loader.Load(); err != nil {
		log.Fatal(err)
	}

	// trunk-ignore(semgrep/go.lang.security.audit.net.use-tls.use-tls): this is intentional
	log.Fatal(http.ListenAndServe(net.JoinHostPort("", strconv.Itoa(cfg.Port)), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n, err := io.Copy(io.Discard, r.Body)
		if err != nil {
			log.Printf("%s %s ERROR %v", r.Method, r.URL, err)
			http.Error(w, fmt.Sprintf("failed to read body: %v", err), http.StatusBadRequest)
			return
		}

		log.Printf("%s %s SUCCESS %d bytes", r.Method, r.URL, n)
	})))
}
