package main

import (
	"bytes"
	"io"
	"os"
	"testing"
	"time"
)

func TestSayHello(t *testing.T) {
	oldStdout := os.Stdout
	defer func() {
		os.Stdout = oldStdout
	}()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("failed to create pipe: %v", err)
	}
	os.Stdout = w

	ch := make(chan string)
	go func() {
		var buf bytes.Buffer
		if _, err := io.Copy(&buf, r); err != nil && err != io.EOF {
			t.Errorf("error during stdout copy: %v", err)
		}
		ch <- buf.String()
	}()

	SayHello()
	if err = w.Close(); err != nil {
		t.Fatalf("error closing stdout pipe")
	}

	var stdoutStr string
	select {
	case stdoutStr = <-ch:
	case <-time.After(1 * time.Second):
		t.Fatal("Timed out waiting for stdout to close")
	}

	expected := "Hello, World!"
	if stdoutStr != expected {
		t.Fatalf("Stdout output %q != %q", stdoutStr, expected)
	}
}
