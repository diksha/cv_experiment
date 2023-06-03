package main

import (
	"fmt"
	"runtime"
)

func main() {
	fmt.Println("hello world")
	fmt.Printf("Go version: %s\n", runtime.Version())
}
